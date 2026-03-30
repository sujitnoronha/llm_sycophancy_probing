#!/usr/bin/env python3
"""
Comprehensive audit of the sycophancy benchmark dataset.
Checks questions.json, personas.json, and conversations.json for
structural integrity, consistency, and correctness.
"""

import json
import sys
from collections import Counter, defaultdict

# ─── Paths ───────────────────────────────────────────────────────────────────
QUESTIONS_PATH  = "/home/ikulkar1/llm_syc/llm_sycophancy_probing/src/dataset/questions.json"
PERSONAS_PATH   = "/home/ikulkar1/llm_syc/llm_sycophancy_probing/src/dataset/personas.json"
CONVERSATIONS_PATH = "/home/ikulkar1/llm_syc/llm_sycophancy_probing/outputs/dataset/conversations.json"

# ─── Counters ────────────────────────────────────────────────────────────────
passes = 0
fails  = 0
warnings = 0

def report(label, passed, detail=""):
    global passes, fails
    tag = "PASS" if passed else "FAIL"
    if not passed:
        fails += 1
    else:
        passes += 1
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")

def warn(label, detail=""):
    global warnings
    warnings += 1
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [WARN] {label}{suffix}")

# ─── Load data ───────────────────────────────────────────────────────────────
print("=" * 80)
print("SYCOPHANCY BENCHMARK DATASET AUDIT")
print("=" * 80)

try:
    with open(QUESTIONS_PATH) as f:
        questions_data = json.load(f)
    questions = questions_data.get("questions", [])
    print(f"\nLoaded questions.json  ({len(questions)} questions)")
except Exception as e:
    print(f"\nFATAL: Cannot load questions.json: {e}")
    sys.exit(1)

try:
    with open(PERSONAS_PATH) as f:
        personas_data = json.load(f)
    print(f"Loaded personas.json")
except Exception as e:
    print(f"\nFATAL: Cannot load personas.json: {e}")
    sys.exit(1)

try:
    with open(CONVERSATIONS_PATH) as f:
        conversations = json.load(f)
    print(f"Loaded conversations.json  ({len(conversations)} conversations)")
except Exception as e:
    print(f"\nFATAL: Cannot load conversations.json: {e}")
    sys.exit(1)

EXPECTED_DOMAINS = [
    "physics", "medicine", "history", "mathematics", "geography",
    "law", "computer_science", "economics", "chemistry", "biology"
]

REQUIRED_FIELDS = [
    "question_id", "domain", "difficulty", "question_text",
    "correct_answer", "acceptable_variants", "wrong_answer",
    "wrong_answer_plausible_reason", "matching_persona_domain"
]

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUESTIONS.JSON AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("1. QUESTIONS.JSON AUDIT")
print("=" * 80)

# --- 1a. Total count ---
print("\n1a. Total question count")
report("Exactly 200 questions", len(questions) == 200,
       f"Found {len(questions)}")

# --- 1b. Domain counts ---
print("\n1b. Domain distribution (expect 20 per domain, 10 domains)")
domain_counts = Counter(q["domain"] for q in questions)
all_domains_present = set(domain_counts.keys()) == set(EXPECTED_DOMAINS)
report("All 10 expected domains present", all_domains_present,
       f"Found domains: {sorted(domain_counts.keys())}")

all_twenty = all(v == 20 for v in domain_counts.values())
report("Exactly 20 questions per domain", all_twenty)
if not all_twenty:
    for d in sorted(domain_counts):
        if domain_counts[d] != 20:
            print(f"      {d}: {domain_counts[d]}")

# --- 1c. Required fields ---
print("\n1c. Required fields present on every question")
missing_fields_map = defaultdict(list)
for q in questions:
    for field in REQUIRED_FIELDS:
        if field not in q:
            missing_fields_map[field].append(q.get("question_id", "UNKNOWN"))
all_fields_ok = len(missing_fields_map) == 0
report("All required fields present", all_fields_ok)
if not all_fields_ok:
    for field, ids in missing_fields_map.items():
        print(f"      Missing '{field}' in: {ids[:5]}{'...' if len(ids)>5 else ''}")

# --- 1d. Duplicate question_ids ---
print("\n1d. Duplicate question_ids")
qids = [q["question_id"] for q in questions]
dup_ids = [qid for qid, cnt in Counter(qids).items() if cnt > 1]
report("No duplicate question_ids", len(dup_ids) == 0,
       f"Duplicates: {dup_ids}" if dup_ids else "")

# --- 1e. Duplicate question_text (exact) ---
print("\n1e. Duplicate question_text (exact match)")
qtexts = [q["question_text"] for q in questions]
dup_texts = [t for t, cnt in Counter(qtexts).items() if cnt > 1]
report("No duplicate question_text", len(dup_texts) == 0,
       f"{len(dup_texts)} duplicates found" if dup_texts else "")
if dup_texts:
    for t in dup_texts[:5]:
        ids_with_text = [q["question_id"] for q in questions if q["question_text"] == t]
        print(f"      Text: \"{t[:80]}...\"  IDs: {ids_with_text}")

# --- 1f. Near-duplicate check (same correct_answer + same domain) ---
print("\n1f. Near-duplicate check (same correct_answer + same domain)")
answer_domain = defaultdict(list)
for q in questions:
    key = (q["correct_answer"].strip().lower(), q["domain"])
    answer_domain[key].append(q["question_id"])
suspicious = {k: v for k, v in answer_domain.items() if len(v) > 1}
report("No near-duplicates (same answer+domain)", len(suspicious) == 0,
       f"{len(suspicious)} suspicious groups" if suspicious else "")
if suspicious:
    for (ans, dom), ids in sorted(suspicious.items()):
        print(f"      Domain={dom}, Answer=\"{ans}\" -> {ids}")

# --- 1g. Difficulty validation ---
print("\n1g. Difficulty field validation")
bad_diff = [(q["question_id"], q.get("difficulty"))
            for q in questions if q.get("difficulty") not in VALID_DIFFICULTIES]
report("All difficulties in {easy, medium, hard}", len(bad_diff) == 0,
       f"Invalid: {bad_diff}" if bad_diff else "")

# --- 1h. matching_persona_domain == domain ---
print("\n1h. matching_persona_domain matches domain")
mismatch_persona = [(q["question_id"], q["domain"], q["matching_persona_domain"])
                    for q in questions
                    if q.get("matching_persona_domain") != q.get("domain")]
report("matching_persona_domain == domain for all", len(mismatch_persona) == 0,
       f"{len(mismatch_persona)} mismatches" if mismatch_persona else "")
if mismatch_persona:
    for qid, dom, mpd in mismatch_persona[:5]:
        print(f"      {qid}: domain={dom}, matching_persona_domain={mpd}")

# --- 1i. Difficulty distribution per domain ---
print("\n1i. Difficulty distribution per domain")
diff_by_domain = defaultdict(lambda: Counter())
for q in questions:
    diff_by_domain[q["domain"]][q["difficulty"]] += 1
print("      Domain              easy  medium  hard  total")
print("      " + "-" * 50)
for dom in sorted(EXPECTED_DOMAINS):
    c = diff_by_domain[dom]
    total = sum(c.values())
    print(f"      {dom:<20} {c.get('easy',0):>4}  {c.get('medium',0):>6}  {c.get('hard',0):>4}  {total:>5}")

# --- 1j. Empty / very short fields ---
print("\n1j. Empty or very short fields")
short_issues = []
for q in questions:
    qid = q["question_id"]
    if len(q.get("correct_answer", "").strip()) < 1:
        short_issues.append((qid, "correct_answer", repr(q.get("correct_answer"))))
    if len(q.get("question_text", "").strip()) < 1:
        short_issues.append((qid, "question_text", repr(q.get("question_text"))))
    if len(q.get("wrong_answer", "").strip()) < 1:
        short_issues.append((qid, "wrong_answer", repr(q.get("wrong_answer"))))
    if len(q.get("wrong_answer_plausible_reason", "").strip()) < 1:
        short_issues.append((qid, "wrong_answer_plausible_reason", repr(q.get("wrong_answer_plausible_reason"))))
    if not isinstance(q.get("acceptable_variants"), list) or len(q.get("acceptable_variants", [])) == 0:
        short_issues.append((qid, "acceptable_variants", "empty or not a list"))

report("No empty / very short critical fields", len(short_issues) == 0,
       f"{len(short_issues)} issues" if short_issues else "")
if short_issues:
    for qid, field, val in short_issues[:10]:
        print(f"      {qid}: {field} = {val}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERSONAS.JSON AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. PERSONAS.JSON AUDIT")
print("=" * 80)

authority_personas = personas_data.get("authority_personas", {})

# --- 2a. All 10 domains in high/medium/low ---
print("\n2a. All 10 domains present in high, medium, low authority levels")
for level in ["high", "medium", "low"]:
    level_data = authority_personas.get(level, {})
    if level == "low":
        # low-authority personas may also be domain-keyed
        domains_in_level = set(level_data.keys()) if isinstance(level_data, dict) else set()
    else:
        domains_in_level = set(level_data.keys()) if isinstance(level_data, dict) else set()
    missing = set(EXPECTED_DOMAINS) - domains_in_level
    report(f"'{level}' has all 10 domains", len(missing) == 0,
           f"Missing: {missing}" if missing else "")

# --- 2b. Persona field checks (prefix, reinforcement, control_prefix non-empty) ---
print("\n2b. Persona fields non-empty (prefix, reinforcement, control_prefix)")
persona_field_issues = []
for level in ["high", "medium", "low"]:
    level_data = authority_personas.get(level, {})
    if not isinstance(level_data, dict):
        continue
    for domain, pdata in level_data.items():
        for pfield in ["prefix", "reinforcement", "control_prefix"]:
            val = pdata.get(pfield)
            if not isinstance(val, str) or len(val.strip()) == 0:
                persona_field_issues.append((level, domain, pfield, repr(val)))
report("All persona fields non-empty strings", len(persona_field_issues) == 0,
       f"{len(persona_field_issues)} issues" if persona_field_issues else "")
if persona_field_issues:
    for lv, dom, fld, val in persona_field_issues:
        print(f"      {lv}/{dom}: {fld} = {val}")

# --- 2c. domain_mismatch_rotation ---
print("\n2c. domain_mismatch_rotation validation")
rotation = personas_data.get("domain_mismatch_rotation", {})
rot_sources = set(rotation.keys())
rot_targets = set(rotation.values())

report("Rotation covers all 10 domains as sources",
       rot_sources == set(EXPECTED_DOMAINS),
       f"Sources: {sorted(rot_sources)}")

report("Rotation covers all 10 domains as targets",
       rot_targets == set(EXPECTED_DOMAINS),
       f"Targets: {sorted(rot_targets)}")

# No domain maps to itself
self_maps = [d for d, t in rotation.items() if d == t]
report("No domain maps to itself", len(self_maps) == 0,
       f"Self-maps: {self_maps}" if self_maps else "")

# Valid permutation (each appears exactly once as source and target)
source_counts = Counter(rotation.keys())
target_counts = Counter(rotation.values())
perm_ok = (all(v == 1 for v in source_counts.values()) and
           all(v == 1 for v in target_counts.values()) and
           len(rotation) == 10)
report("Rotation is a valid permutation", perm_ok)

print("\n      Rotation mapping:")
for src in sorted(rotation):
    print(f"        {src:>20} -> {rotation[src]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONVERSATIONS.JSON AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. CONVERSATIONS.JSON AUDIT")
print("=" * 80)

# --- 3a. Total count ---
print("\n3a. Total conversation count")
report("Exactly 3200 conversations", len(conversations) == 3200,
       f"Found {len(conversations)}")

# --- 3b. Domain distribution ---
print("\n3b. Domain distribution (expect 320 each)")
conv_domain_counts = Counter(c["domain"] for c in conversations)
all_320 = all(conv_domain_counts.get(d) == 320 for d in EXPECTED_DOMAINS)
report("All 10 domains with 320 conversations each", all_320)
if not all_320:
    for d in sorted(EXPECTED_DOMAINS):
        cnt = conv_domain_counts.get(d, 0)
        flag = " <--" if cnt != 320 else ""
        print(f"      {d}: {cnt}{flag}")

# --- 3c. All 16 conditions per question ---
print("\n3c. All 16 conditions represented for each question")
conditions_defined = [c["id"] for c in personas_data.get("conditions", [])]
conditions_per_q = defaultdict(set)
for c in conversations:
    conditions_per_q[c["question_id"]].add(c["condition"]["id"])

missing_conditions = {}
for qid, conds in conditions_per_q.items():
    missing = set(conditions_defined) - conds
    extra = conds - set(conditions_defined)
    if missing or extra:
        missing_conditions[qid] = {"missing": missing, "extra": extra}

report("Every question has all 16 conditions",
       len(missing_conditions) == 0,
       f"{len(missing_conditions)} questions with issues" if missing_conditions else "")
if missing_conditions:
    for qid, info in list(missing_conditions.items())[:5]:
        if info["missing"]:
            print(f"      {qid}: MISSING conditions: {info['missing']}")
        if info["extra"]:
            print(f"      {qid}: EXTRA conditions: {info['extra']}")

# --- 3d. No duplicate conversation_ids ---
print("\n3d. Duplicate conversation_ids")
conv_ids = [c["conversation_id"] for c in conversations]
dup_conv_ids = [cid for cid, cnt in Counter(conv_ids).items() if cnt > 1]
report("No duplicate conversation_ids", len(dup_conv_ids) == 0,
       f"{len(dup_conv_ids)} duplicates" if dup_conv_ids else "")
if dup_conv_ids:
    for cid in dup_conv_ids[:5]:
        print(f"      {cid}")

# --- 3e. turn1_user and turn3_user non-empty ---
print("\n3e. turn1_user and turn3_user are non-empty")
empty_turns = []
for c in conversations:
    cid = c["conversation_id"]
    if not c.get("turn1_user") or len(str(c["turn1_user"]).strip()) == 0:
        empty_turns.append((cid, "turn1_user"))
    if not c.get("turn3_user") or len(str(c["turn3_user"]).strip()) == 0:
        empty_turns.append((cid, "turn3_user"))
report("All turn1_user and turn3_user non-empty", len(empty_turns) == 0,
       f"{len(empty_turns)} empty turns" if empty_turns else "")
if empty_turns:
    for cid, field in empty_turns[:5]:
        print(f"      {cid}: {field} is empty")

# --- 3f. turn2_assistant and turn4_assistant are null ---
print("\n3f. turn2_assistant and turn4_assistant are null (unfilled)")
nonnull_assistants = []
for c in conversations:
    cid = c["conversation_id"]
    if c.get("turn2_assistant") is not None:
        nonnull_assistants.append((cid, "turn2_assistant"))
    if c.get("turn4_assistant") is not None:
        nonnull_assistants.append((cid, "turn4_assistant"))
report("All turn2/turn4_assistant are null", len(nonnull_assistants) == 0,
       f"{len(nonnull_assistants)} non-null" if nonnull_assistants else "")
if nonnull_assistants:
    for cid, field in nonnull_assistants[:5]:
        print(f"      {cid}: {field} is not null")

# --- 3g. Matched/mismatched domain logic ---
print("\n3g. Domain match/mismatch logic")
match_issues = []
for c in conversations:
    cid = c["conversation_id"]
    dm = c["condition"].get("domain_match")
    d = c["domain"]
    pd = c.get("persona_domain")
    if dm == "matched" and pd != d:
        match_issues.append((cid, "matched but persona_domain != domain",
                             f"domain={d}, persona_domain={pd}"))
    elif dm == "mismatched" and pd == d:
        match_issues.append((cid, "mismatched but persona_domain == domain",
                             f"domain={d}, persona_domain={pd}"))
report("Matched => persona_domain==domain; Mismatched => persona_domain!=domain",
       len(match_issues) == 0,
       f"{len(match_issues)} issues" if match_issues else "")
if match_issues:
    for cid, issue, detail in match_issues[:10]:
        print(f"      {cid}: {issue} ({detail})")

# --- 3h. wrong_answer appears in turn3_user (spot check all) ---
print("\n3h. wrong_answer appears in turn3_user")
wrong_answer_missing = []
for c in conversations:
    wa = c.get("wrong_answer", "")
    t3 = c.get("turn3_user", "")
    if wa and wa not in t3:
        wrong_answer_missing.append((c["conversation_id"], wa, t3[:80]))
report("wrong_answer appears in turn3_user for all conversations",
       len(wrong_answer_missing) == 0,
       f"{len(wrong_answer_missing)} missing" if wrong_answer_missing else "")
if wrong_answer_missing:
    for cid, wa, t3 in wrong_answer_missing[:5]:
        print(f"      {cid}: wrong_answer=\"{wa}\" NOT found in turn3: \"{t3}...\"")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. CROSS-CHECKS")
print("=" * 80)

# --- 4a. Every question_id in conversations exists in questions ---
print("\n4a. Every conversation question_id exists in questions.json")
question_ids_set = set(q["question_id"] for q in questions)
conv_qids = set(c["question_id"] for c in conversations)
orphan_conv = conv_qids - question_ids_set
report("All conversation question_ids found in questions.json",
       len(orphan_conv) == 0,
       f"Orphans: {orphan_conv}" if orphan_conv else "")

# Also check reverse: every question_id in questions appears in conversations
orphan_q = question_ids_set - conv_qids
report("All question_ids in questions.json appear in conversations.json",
       len(orphan_q) == 0,
       f"Unused questions: {orphan_q}" if orphan_q else "")

# --- 4b. Every domain in questions has personas ---
print("\n4b. Every domain in questions.json has personas in personas.json")
persona_domains = set()
for level in ["high", "medium", "low"]:
    level_data = authority_personas.get(level, {})
    if isinstance(level_data, dict):
        persona_domains.update(level_data.keys())
question_domains = set(q["domain"] for q in questions)
missing_persona_domains = question_domains - persona_domains
report("All question domains covered by personas",
       len(missing_persona_domains) == 0,
       f"Missing: {missing_persona_domains}" if missing_persona_domains else "")

# --- 4c. Correct answer / wrong answer consistency between questions and conversations ---
print("\n4c. Answer consistency between questions.json and conversations.json")
q_lookup = {q["question_id"]: q for q in questions}
answer_mismatches = []
for c in conversations:
    qid = c["question_id"]
    if qid in q_lookup:
        q = q_lookup[qid]
        if c.get("correct_answer") != q["correct_answer"]:
            answer_mismatches.append((c["conversation_id"], "correct_answer",
                                     q["correct_answer"], c.get("correct_answer")))
        if c.get("wrong_answer") != q["wrong_answer"]:
            answer_mismatches.append((c["conversation_id"], "wrong_answer",
                                     q["wrong_answer"], c.get("wrong_answer")))
report("Correct/wrong answers match between questions and conversations",
       len(answer_mismatches) == 0,
       f"{len(answer_mismatches)} mismatches" if answer_mismatches else "")
if answer_mismatches:
    for cid, field, expected, got in answer_mismatches[:5]:
        print(f"      {cid}: {field} expected=\"{expected}\", got=\"{got}\"")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)
total = passes + fails
print(f"\n  Total checks:  {total}")
print(f"  PASSED:        {passes}")
print(f"  FAILED:        {fails}")
print(f"  WARNINGS:      {warnings}")
if fails == 0:
    print("\n  >>> ALL CHECKS PASSED <<<")
else:
    print(f"\n  >>> {fails} CHECK(S) FAILED -- REVIEW ABOVE <<<")
print()
