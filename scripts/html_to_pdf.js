const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
  const root = path.resolve(__dirname, '..');
  const inFile = path.join(root, 'docs', 'ML_PIPELINE.html');
  const outFile = path.join(root, 'docs', 'ML_PIPELINE.pdf');
  if (!fs.existsSync(inFile)) {
    console.error('Input HTML not found:', inFile);
    process.exit(2);
  }

  const browser = await puppeteer.launch({args: ['--no-sandbox', '--disable-setuid-sandbox']});
  const page = await browser.newPage();
  await page.goto('file://' + inFile, {waitUntil: 'networkidle2'});
  // give a short pause for SVG rendering/animations to settle (compatible)
  await new Promise((res) => setTimeout(res, 500));
  // save a screenshot for debugging visual output
  const screenshotPath = path.join(root, 'docs', 'ML_PIPELINE_preview.png');
  await page.screenshot({path: screenshotPath, fullPage: true});
  await page.pdf({path: outFile, format: 'A4', printBackground: true});
  await browser.close();
  console.log('Wrote PDF to', outFile);
})();
