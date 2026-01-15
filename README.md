# TEMPO Analyzer

A modern desktop application for analyzing NASA TEMPO satellite data (NO₂ and HCHO) with smart storage and beautiful visualizations.

![TEMPO Analyzer](docs/screenshot.png)

## Features

- 📊 **Dataset Creator** - Download and configure TEMPO data with custom regions, dates, and quality filters
- 🗺️ **Interactive Maps** - Visualize NO₂, HCHO, and FNR (Formaldehyde-to-NO₂ Ratio) on geographic maps
- 🔍 **Data Inspector** - Browse individual granules and see exactly what was downloaded
- 📁 **Smart Storage** - Automatic deduplication and resume support for interrupted downloads
- 📤 **Excel Export** - Multiple export formats for data analysis
- 🎨 **Modern UI** - Beautiful Material Design 3 interface built with Flet

## Installation

### Option 1: Standalone Executable (Recommended)

Download the latest release from the [Releases](https://github.com/your-repo/tempo-analyzer/releases) page.

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/your-repo/tempo-analyzer.git
cd tempo-analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m tempo_app.main
```

## Quick Start

1. **Create a Dataset**: Select a region (e.g., "Southern California"), date range, and quality filters
2. **Download**: Click "Download & Create Dataset" - the app will fetch data from NASA RSIG
3. **Visualize**: Use the Compare tab to generate side-by-side maps
4. **Export**: Export data to Excel for further analysis

## System Requirements

- Windows 10/11, macOS 10.15+, or Linux
- 4 GB RAM minimum (8 GB recommended)
- 2 GB free disk space for application and cache

## Data Sources

- **TEMPO Satellite Data**: Downloaded via [NASA RSIG API](https://www.epa.gov/rsig)
- **Road Networks**: US Census Bureau TIGER/Line shapefiles (auto-downloaded)

## License

MIT License - see [LICENSE](LICENSE) for details.
