"""
Home Affairs web scraper — Phase 2 (Weeks 3-4).

Will implement:
- Scraping of immi.homeaffairs.gov.au 485 visa pages
- PDF download with Last-Modified checking
- HTML content parsing and normalization
- Rate limiting and robots.txt compliance
- Caching of HTML snapshots for change detection
"""


class HomeAffairsScraper:
    """Scraper for Australian immigration website."""

    def __init__(self):
        raise NotImplementedError("Phase 2 — Coming in Weeks 3-4")

    def scrape_page(self, url: str) -> dict:
        """Scrape a single page and return structured content."""
        raise NotImplementedError

    def download_pdf(self, url: str, output_path: str) -> str:
        """Download a PDF file if it has been modified."""
        raise NotImplementedError

    def run_daily_scrape(self) -> list[dict]:
        """Execute the daily scraping routine for all monitored URLs."""
        raise NotImplementedError