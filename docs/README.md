# Research sharing page

This buildless site collects the supplied paper, the 12-minute BayLearn presentation, and the research repository in one place. The page preserves Sujit Noronha as first author and Steven Au as the talk presenter. The manuscript remains labeled as the COLM 2026 draft under review.

## Enable GitHub Pages

After merging this page into `main`, a repository administrator can enable it:

1. Open **Settings → Pages**.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Select **main** and **/docs**, then **Save**.

The expected URL is https://sujitnoronha.github.io/llm_sycophancy_probing/.

The included PNG and SVG QR codes encode this exact URL. It becomes usable after GitHub Pages finishes publishing. No custom domain is required.

Official instructions: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site.

## Update materials

Replace `assets/blind-deference-paper.pdf` or `assets/baylearn-slides.pptx` while keeping their filenames. Existing links and the QR code will continue to work. If the page URL changes, regenerate the QR codes and update the canonical URL in `index.html` and `site.js`.

The PDF is the exact manuscript supplied for sharing. `docs/.gitignore` makes a narrow exception to the repository's general PDF exclusion.

## Local preview

Serve `docs` with any static HTTP server, for example `python -m http.server 8080 --directory docs`. The site needs no build step, external fonts, runtime dependencies, analytics, or accounts.
