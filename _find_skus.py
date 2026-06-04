"""Find Bluetooth speaker SKUs from Best Buy search page using Playwright."""
from playwright.sync_api import sync_playwright
import re, json

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    )
    page = context.new_page()

    print("Loading search page...")
    page.goto(
        "https://www.bestbuy.com/site/searchpage.jsp?st=bluetooth+speaker&cp=1",
        wait_until="networkidle",
        timeout=45000,
    )

    content = page.content()
    print(f"Rendered page length: {len(content)} chars")

    # Extract SKUs from HTML
    skus = set()
    for m in re.finditer(r'data-sku-id="(\d+)"', content):
        skus.add(m.group(1))
    for m in re.finditer(r'"skuId":"?(\d+)"?', content):
        skus.add(m.group(1))

    print(f"\nUnique SKUs from HTML: {len(skus)}")
    for s in sorted(list(skus))[:40]:
        print(f"  {s}")

    # Try getting elements via JS
    print("\nProbing page JS...")
    script_data = page.evaluate("""() => {
        const results = [];
        const selectors = ['[data-sku-id]', '.sku-item', '[class*="product"]'];
        for (const sel of selectors) {
            const els = document.querySelectorAll(sel);
            if (els.length > 0) {
                results.push({selector: sel, count: els.length});
                // sample first 2
                els.forEach((el, i) => {
                    if (i < 2) {
                        results.push({
                            tag: el.tagName,
                            id: el.id,
                            classes: el.className.slice(0, 100),
                            data_sku: el.getAttribute('data-sku-id'),
                            text: (el.textContent || '').slice(0, 80),
                        });
                    }
                });
            }
        }
        return results;
    }""")

    print(f"Probe results: {json.dumps(script_data, indent=2)[:2000]}")
    browser.close()
