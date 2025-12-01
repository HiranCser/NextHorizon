const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
  const root = path.resolve(__dirname, '..');
  const mmdPath = path.join(root, 'docs', 'diagrams', 'architecture.mmd');
  const outPath = path.join(root, 'docs', 'diagrams', 'architecture.svg');
  const mmd = fs.readFileSync(mmdPath, 'utf8');

  const html = `<!doctype html>
  <html>
  <head>
    <meta charset="utf-8" />
    <script src="https://unpkg.com/mermaid@10/dist/mermaid.min.js"></script>
    <style>body{margin:0;padding:10px;font-family:Arial}</style>
  </head>
  <body>
    <div id="container">
      <div class="mermaid">${mmd}</div>
    </div>
    <script>
      mermaid.initialize({startOnLoad:true, theme: 'default'});
    </script>
  </body>
  </html>`;

  const browser = await puppeteer.launch({args: ['--no-sandbox','--disable-setuid-sandbox']});
  const page = await browser.newPage();
  await page.setContent(html, {waitUntil: 'networkidle2'});
  // wait for mermaid to render
  await page.waitForSelector('svg', {timeout: 5000});
  const svg = await page.$eval('svg', node => node.outerHTML);
  fs.writeFileSync(outPath, svg, 'utf8');
  await browser.close();
  console.log('Wrote', outPath);
})();
