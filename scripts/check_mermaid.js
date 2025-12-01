const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
  const root = path.resolve(__dirname, '..');
  const mmdPath = path.join(root, 'docs', 'diagrams', 'architecture.mmd');
  const mmd = fs.readFileSync(mmdPath, 'utf8');
  const html = `<!doctype html>
  <html>
  <head>
    <meta charset="utf-8" />
    <script src="https://unpkg.com/mermaid@10/dist/mermaid.min.js"></script>
  </head>
  <body>
    <script>
      try {
        const txt = ${JSON.stringify(mmd)};
        const parsed = mermaid.parse(txt);
        console.log('PARSE_OK');
      } catch (err) {
        console.log('PARSE_ERROR');
        console.log(err && err.str || err.message || String(err));
      }
    </script>
  </body>
  </html>`;

  const browser = await puppeteer.launch({args: ['--no-sandbox','--disable-setuid-sandbox']});
  const page = await browser.newPage();
  page.on('console', msg => console.log('[PAGE]', msg.text()));
  await page.setContent(html, {waitUntil: 'networkidle2'});
  await browser.close();
})();
