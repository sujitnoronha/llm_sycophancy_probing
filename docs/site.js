'use strict';
const copyButton = document.getElementById('copy-link');
const copyStatus = document.getElementById('copy-status');
const researchUrl = 'https://sujitnoronha.github.io/llm_sycophancy_probing/';
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(researchUrl);
    copyStatus.textContent = 'Link copied.';
  } catch {
    copyStatus.textContent = researchUrl;
  }
});
