# Natural Language Chart Generation for TEMPO Analyzer - Implementation Plan

## Executive Summary

Add a hybrid NL-to-chart system that uses **Gemini API as a query interpreter** (not chart generator) to translate natural language into structured parameters, then **Matplotlib generates publication-quality charts locally**. This preserves scientific reproducibility while adding user-friendly NL interface.

## Architecture Overview

```
User NL Query ("Show NO₂ trends at BV during weekdays, cloud < 0.3")
    ↓
Gemini API (query parser) → Structured JSON parameters
    ↓
Local Parameter Validator (checks against DB)
    ↓ (if ambiguous)
Interactive Clarification Dialog
    ↓
Matplotlib Chart Generator (local, reproducible)
    ↓
Outputs: PNG/SVG chart + Python script + metadata JSON
```

## Why This Hybrid Approach Works

### Addresses Scientific Requirements
- **Reproducibility**: Matplotlib code saved alongside charts, can regenerate without AI
- **Precision**: Full control over axes, scales, formatting via Matplotlib
- **Data Privacy**: Only query text sent to API, not actual atmospheric data
- **Performance**: Chart rendering is local and fast
- **Offline Capability**: Saved scripts work without internet

### Adds User Value
- **Natural language**: "Show diurnal NO₂ patterns at Bountiful for August weekdays"
- **Smart interpretation**: Gemini maps "Bountiful" → site "BV", "August" → date range
- **Guided refinement**: AI asks clarifying questions when ambiguous
- **Discoverability**: Users don't need to learn complex UI

## Critical Implementation Decisions

### 1. Data Flow: Text Only to API ✓
- **Send to Gemini**: Query text + available dataset/site metadata (IDs, names, date ranges)
- **Never send**: Actual NO₂/HCHO measurements, NetCDF data, proprietary research data
- **Gemini returns**: Structured JSON with chart parameters

### 2. Reproducibility: Save Everything ✓
For each generated chart, save:
```
charts/
  chart_20260115_143022/
    chart.png              # Visual output
    chart.svg              # Vector version
    parameters.json        # Structured params from Gemini
    generate_chart.py      # Matplotlib code to regenerate
    query.txt              # Original NL query
```

This allows:
- Regenerating exact chart for publications
- Debugging if chart looks wrong
- Tweaking parameters manually
- Sharing reproducible workflows

### 3. Gemini Output Schema
Gemini must return strict JSON schema:
```json
{
  "chart_type": "time_series | diurnal_cycle | correlation | distribution | comparison",
  "datasets": [{"dataset_id": "uuid", "label": "August 2024"}],
  "sites": [{"site_id": "BV", "name": "Bountiful"}],
  "variables": ["NO2", "HCHO", "FNR"],
  "date_range": {"start": "2024-08-01", "end": "2024-08-31"},
  "filters": {
    "days_of_week": [1,2,3,4,5],
    "hours_utc": [14,15,16,17,18],
    "cloud_fraction_max": 0.3,
    "sza_max": 60
  },
  "aggregation": "hourly | daily | weekly",
  "styling": {
    "title": "NO₂ Trends at Bountiful (Weekdays, Low Cloud)",
    "xlabel": "Date",
    "ylabel": "NO₂ (molecules/cm²)"
  },
  "clarifications_needed": [] or ["Which aggregation: hourly or daily?"]
}
```

### 4. Error Handling & Clarifications
When Gemini identifies ambiguities, return:
```json
{
  "status": "needs_clarification",
  "clarifications_needed": [
    {
      "question": "Which site did you mean?",
      "options": ["BV (Bountiful, UT)", "HC (Herriman, UT)", "All sites"],
      "context": "Found multiple sites in the selected region"
    }
  ]
}
```

UI shows interactive dialog, user selects, query re-sent with clarification.

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Create Chart Generator Module
**File**: `src/tempo_app/core/chart_generator.py`

Implement Matplotlib-based chart generation:
- `TimeSeriesChart`: Plot variable(s) over time at specific sites
- `DiurnalCycleChart`: Show hourly patterns (box plots or line plots by hour)
- `CorrelationChart`: Scatter plots of var1 vs var2
- `DistributionChart`: Histograms or KDE plots
- `ComparisonChart`: Multi-site or multi-dataset overlays

Each returns:
- Chart image path (PNG/SVG)
- Generated Python code (as string)
- Metadata dict

**Key Implementation Details**:
```python
class ChartGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_time_series(
        self,
        data: xr.Dataset,
        variable: str,
        sites: List[Site],
        title: str,
        **styling_kwargs
    ) -> ChartResult:
        """
        Generate time series chart.

        Returns:
            ChartResult(
                png_path=Path,
                svg_path=Path,
                script_code=str,  # Python code to regenerate
                metadata=dict
            )
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot data for each site
        for site in sites:
            site_data = data.sel(lat=site.lat, lon=site.lon, method='nearest')
            ax.plot(site_data.time, site_data[variable], label=site.name)

        # Styling
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(self._get_variable_label(variable))
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save outputs
        chart_id = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        chart_dir = self.output_dir / chart_id
        chart_dir.mkdir(exist_ok=True)

        png_path = chart_dir / "chart.png"
        svg_path = chart_dir / "chart.svg"

        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.close(fig)

        # Generate reproducible script
        script_code = self._generate_script(
            chart_type='time_series',
            data_query=...,  # How to recreate data
            plotting_code=...,  # Exact plotting commands
        )

        script_path = chart_dir / "generate_chart.py"
        script_path.write_text(script_code)

        return ChartResult(
            png_path=png_path,
            svg_path=svg_path,
            script_path=script_path,
            script_code=script_code,
            metadata={
                'chart_id': chart_id,
                'chart_type': 'time_series',
                'variable': variable,
                'sites': [s.id for s in sites],
                'created_at': datetime.now().isoformat()
            }
        )

    def _generate_script(self, chart_type, data_query, plotting_code):
        """Generate standalone Python script that recreates chart."""
        return f"""
#!/usr/bin/env python3
\"\"\"
Auto-generated script to reproduce chart.
Generated at: {datetime.now().isoformat()}
Chart type: {chart_type}
\"\"\"

import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

# Load data
{data_query}

# Generate chart
{plotting_code}

# Save
plt.savefig('chart_regenerated.png', dpi=300, bbox_inches='tight')
plt.savefig('chart_regenerated.svg', format='svg', bbox_inches='tight')
print("Chart regenerated successfully")
"""
```

#### 1.2 Create Gemini Query Interpreter
**File**: `src/tempo_app/integrations/gemini_client.py`

Responsibilities:
- Send NL query + context (available datasets, sites) to Gemini
- Parse JSON response
- Handle API errors (rate limits, network issues)
- Validate response schema
- Return structured parameters or clarification requests

**Implementation**:
```python
import google.generativeai as genai
from typing import Dict, List, Optional
import json
from dataclasses import dataclass

@dataclass
class ChartParameters:
    chart_type: str
    datasets: List[Dict]
    sites: List[Dict]
    variables: List[str]
    date_range: Dict
    filters: Dict
    aggregation: str
    styling: Dict
    clarifications_needed: List[Dict]

class GeminiQueryInterpreter:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.cache = {}  # Query cache to reduce API calls

    def interpret_query(
        self,
        query: str,
        context: Dict
    ) -> ChartParameters:
        """
        Interpret natural language query into structured parameters.

        Args:
            query: User's natural language query
            context: Available datasets, sites, variables

        Returns:
            ChartParameters with parsed values or clarifications needed
        """
        # Check cache
        cache_key = f"{query}:{hash(json.dumps(context, sort_keys=True))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        # Call Gemini
        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={
                    'temperature': 0.1,  # Low temp for consistency
                    'response_mime_type': 'application/json'
                }
            )

            # Parse response
            params_dict = json.loads(response.text)
            params = ChartParameters(**params_dict)

            # Cache result
            self.cache[cache_key] = params

            return params

        except Exception as e:
            # Handle API errors
            return ChartParameters(
                chart_type='error',
                clarifications_needed=[{
                    'question': f'Error interpreting query: {str(e)}',
                    'options': ['Try rephrasing your request'],
                    'context': 'API error'
                }]
            )

    def _build_system_prompt(self) -> str:
        return """You are a query interpreter for a NASA TEMPO atmospheric data analyzer.
Parse natural language chart requests into structured JSON parameters.

Available chart types:
- time_series: Plot variable(s) over time
- diurnal_cycle: Show patterns by hour of day (0-23 UTC)
- correlation: Scatter plot of two variables
- distribution: Histogram of variable values
- comparison: Overlay multiple sites/datasets

Variables:
- NO2: Nitrogen dioxide tropospheric column (molecules/cm²)
- HCHO: Formaldehyde total column (molecules/cm²)
- FNR: Formaldehyde-to-NO₂ ratio

CRITICAL RULES:
1. Only use provided dataset/site IDs from context
2. If query is ambiguous, populate clarifications_needed array
3. Return valid JSON only, no explanatory text
4. If request is impossible, explain in clarifications_needed
5. Preserve scientific terminology and units
6. For date ranges, infer from context if not specified
7. Map informal names to formal IDs (e.g., "Bountiful" → site ID "BV")

Return JSON following this exact schema:
{
  "chart_type": "time_series | diurnal_cycle | correlation | distribution | comparison",
  "datasets": [{"dataset_id": "uuid", "label": "August 2024"}],
  "sites": [{"site_id": "BV", "name": "Bountiful"}],
  "variables": ["NO2"],
  "date_range": {"start": "2024-08-01", "end": "2024-08-31"},
  "filters": {
    "days_of_week": [0,1,2,3,4,5,6],  # 0=Monday
    "hours_utc": [0-23],
    "cloud_fraction_max": 1.0,
    "sza_max": 90
  },
  "aggregation": "hourly | daily | weekly",
  "styling": {
    "title": "Chart Title",
    "xlabel": "X Label",
    "ylabel": "Y Label"
  },
  "clarifications_needed": []
}
"""

    def _build_user_prompt(self, query: str, context: Dict) -> str:
        return f"""Available Context:
{json.dumps(context, indent=2)}

User Query: "{query}"

Parse this query and return structured JSON parameters."""
```

#### 1.3 Create Parameter Validator
**File**: `src/tempo_app/core/chart_validator.py`

Validates Gemini output against actual database:
```python
from dataclasses import dataclass
from typing import List, Optional
from src.tempo_app.storage.models import Dataset, Site
from src.tempo_app.integrations.gemini_client import ChartParameters

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ChartParameterValidator:
    def __init__(self, db_session):
        self.db = db_session

    def validate(self, params: ChartParameters) -> ValidationResult:
        """Validate parameters against database."""
        errors = []
        warnings = []

        # Validate chart type
        valid_types = ['time_series', 'diurnal_cycle', 'correlation',
                       'distribution', 'comparison']
        if params.chart_type not in valid_types:
            errors.append(f"Invalid chart type: {params.chart_type}")

        # Validate datasets exist
        for ds in params.datasets:
            dataset = self.db.query(Dataset).filter_by(
                id=ds['dataset_id']
            ).first()
            if not dataset:
                errors.append(f"Dataset not found: {ds['dataset_id']}")
            else:
                # Check date range is within dataset bounds
                if params.date_range:
                    start = params.date_range.get('start')
                    end = params.date_range.get('end')
                    if start < dataset.start_date or end > dataset.end_date:
                        warnings.append(
                            f"Date range extends beyond dataset coverage"
                        )

        # Validate sites exist
        for site in params.sites:
            s = self.db.query(Site).filter_by(id=site['site_id']).first()
            if not s:
                errors.append(f"Site not found: {site['site_id']}")

        # Validate variables
        valid_vars = ['NO2', 'HCHO', 'FNR']
        for var in params.variables:
            if var not in valid_vars:
                errors.append(f"Invalid variable: {var}")

        # Chart-specific validation
        if params.chart_type == 'correlation' and len(params.variables) != 2:
            errors.append("Correlation charts require exactly 2 variables")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

### Phase 2: Storage & History

#### 2.1 Database Schema Updates
Add to `src/tempo_app/storage/models.py`:

```python
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

class GeneratedChart(Base):
    __tablename__ = 'generated_charts'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey('datasets.id'))

    # Original query
    query_text = Column(String, nullable=False)

    # Chart configuration
    chart_type = Column(String, nullable=False)
    parameters_json = Column(JSON, nullable=False)  # Full Gemini output

    # File paths (relative to charts directory)
    image_path = Column(String, nullable=False)  # PNG
    svg_path = Column(String)
    script_path = Column(String, nullable=False)  # .py file

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    regenerated_from = Column(UUID(as_uuid=True), ForeignKey('generated_charts.id'), nullable=True)

    # Relationships
    dataset = relationship('Dataset', back_populates='charts')
    original_chart = relationship('GeneratedChart', remote_side=[id])

# Add to Dataset model
Dataset.charts = relationship('GeneratedChart', back_populates='dataset')
```

#### 2.2 Migration Script
Create Alembic migration:
```bash
alembic revision -m "add_generated_charts_table"
```

### Phase 3: UI Integration

#### 3.1 New Charts Page
**File**: `src/tempo_app/ui/pages/charts.py`

```python
import flet as ft
from src.tempo_app.integrations.gemini_client import GeminiQueryInterpreter
from src.tempo_app.core.chart_validator import ChartParameterValidator
from src.tempo_app.core.chart_generator import ChartGenerator

class ChartsPage:
    def __init__(self, page: ft.Page, db_session):
        self.page = page
        self.db = db_session
        self.gemini = GeminiQueryInterpreter(api_key=self._get_api_key())
        self.validator = ChartParameterValidator(db_session)
        self.generator = ChartGenerator(output_dir=Path('charts'))

        self.query_input = ft.TextField(
            label="Describe the chart you want...",
            multiline=True,
            min_lines=3,
            max_lines=5,
            hint_text="Example: Show NO₂ trends at Bountiful for August weekdays",
            expand=True
        )

        self.submit_btn = ft.ElevatedButton(
            "Generate Chart",
            on_click=self.on_generate_click,
            icon=ft.icons.AUTO_GRAPH
        )

        self.loading = ft.ProgressRing(visible=False)
        self.status_text = ft.Text("", size=14)

        self.chart_display = ft.Container(
            content=ft.Text("Your chart will appear here"),
            bgcolor=ft.colors.SURFACE_VARIANT,
            border_radius=8,
            padding=20,
            expand=True
        )

        self.examples = self._build_examples()
        self.history = self._build_history_sidebar()

    async def on_generate_click(self, e):
        """Handle chart generation."""
        query = self.query_input.value
        if not query:
            self.status_text.value = "Please enter a query"
            await self.page.update_async()
            return

        # Show loading state
        self.loading.visible = True
        self.submit_btn.disabled = True
        self.status_text.value = "Interpreting your request..."
        await self.page.update_async()

        try:
            # Get context from database
            context = self._build_context()

            # Interpret query with Gemini
            params = await self._async_interpret(query, context)

            # Check for clarifications
            if params.clarifications_needed:
                await self._show_clarification_dialog(params)
                return

            # Validate parameters
            self.status_text.value = "Validating parameters..."
            await self.page.update_async()

            validation = self.validator.validate(params)
            if not validation.is_valid:
                self.status_text.value = "Errors: " + "; ".join(validation.errors)
                return

            # Generate chart
            self.status_text.value = "Generating chart..."
            await self.page.update_async()

            chart_result = await self._async_generate_chart(params)

            # Display chart
            await self._display_chart(chart_result)

            # Save to database
            self._save_chart_to_db(query, params, chart_result)

            self.status_text.value = "Chart generated successfully!"

        except Exception as ex:
            self.status_text.value = f"Error: {str(ex)}"

        finally:
            self.loading.visible = False
            self.submit_btn.disabled = False
            await self.page.update_async()

    async def _show_clarification_dialog(self, params):
        """Show interactive dialog for clarifications."""
        # Build dialog with questions
        questions = []
        for clarification in params.clarifications_needed:
            question_text = ft.Text(clarification['question'], weight=ft.FontWeight.BOLD)
            options = ft.RadioGroup(
                content=ft.Column([
                    ft.Radio(value=opt, label=opt)
                    for opt in clarification['options']
                ])
            )
            questions.append(ft.Column([question_text, options]))

        dialog = ft.AlertDialog(
            title=ft.Text("Need More Information"),
            content=ft.Column(questions, tight=True, scroll=ft.ScrollMode.AUTO),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog()),
                ft.TextButton("Submit", on_click=lambda e: self._resubmit_with_clarifications())
            ]
        )

        self.page.dialog = dialog
        dialog.open = True
        await self.page.update_async()

    def _build_examples(self):
        """Build example queries panel."""
        examples = [
            "Show NO₂ time series at Bountiful for August 2024",
            "Compare weekday vs weekend HCHO patterns at all sites",
            "Plot NO₂ vs HCHO correlation when cloud fraction < 0.2",
            "Show diurnal cycles for FNR during summer weekends",
            "Distribution of NO₂ values at BV site for rush hours (6-9am, 4-7pm)"
        ]

        return ft.ExpansionTile(
            title=ft.Text("Example Queries"),
            subtitle=ft.Text("Click to use"),
            controls=[
                ft.ListTile(
                    title=ft.Text(ex),
                    on_click=lambda e, example=ex: self._use_example(example)
                )
                for ex in examples
            ]
        )

    def _build_history_sidebar(self):
        """Build chart history sidebar."""
        # Load recent charts from database
        recent_charts = self.db.query(GeneratedChart).order_by(
            GeneratedChart.created_at.desc()
        ).limit(10).all()

        return ft.Container(
            width=250,
            content=ft.Column([
                ft.Text("Recent Charts", size=18, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                *[
                    ft.ListTile(
                        title=ft.Text(chart.query_text, max_lines=2),
                        subtitle=ft.Text(chart.created_at.strftime("%Y-%m-%d %H:%M")),
                        on_click=lambda e, c=chart: self._load_chart(c)
                    )
                    for chart in recent_charts
                ]
            ], scroll=ft.ScrollMode.AUTO),
            bgcolor=ft.colors.SURFACE_VARIANT,
            padding=10
        )

    def build(self):
        """Build the page layout."""
        return ft.Row([
            # Main content
            ft.Container(
                content=ft.Column([
                    ft.Text("Chart Generation", size=24, weight=ft.FontWeight.BOLD),
                    self.examples,
                    ft.Row([
                        self.query_input,
                        ft.Column([
                            self.submit_btn,
                            self.loading
                        ])
                    ]),
                    self.status_text,
                    self.chart_display
                ], expand=True, scroll=ft.ScrollMode.AUTO),
                expand=True,
                padding=20
            ),
            # History sidebar
            self.history
        ], expand=True)
```

#### 3.2 Clarification Dialog Component
**File**: `src/tempo_app/ui/components/clarification_dialog.py`

```python
import flet as ft
from typing import List, Dict, Callable

class ClarificationDialog:
    def __init__(self, clarifications: List[Dict], on_submit: Callable):
        self.clarifications = clarifications
        self.on_submit = on_submit
        self.selections = {}

    def build(self) -> ft.AlertDialog:
        """Build interactive clarification dialog."""
        question_controls = []

        for idx, clarification in enumerate(self.clarifications):
            question_text = ft.Text(
                clarification['question'],
                size=16,
                weight=ft.FontWeight.BOLD
            )

            context = ft.Text(
                clarification.get('context', ''),
                size=12,
                color=ft.colors.SECONDARY
            )

            # Radio group for options
            radio_group = ft.RadioGroup(
                content=ft.Column([
                    ft.Radio(value=opt, label=opt)
                    for opt in clarification['options']
                ]),
                on_change=lambda e, i=idx: self._on_selection(i, e.control.value)
            )

            question_controls.append(
                ft.Container(
                    content=ft.Column([
                        question_text,
                        context,
                        radio_group
                    ]),
                    padding=10,
                    border=ft.border.all(1, ft.colors.OUTLINE),
                    border_radius=8,
                    margin=ft.margin.only(bottom=10)
                )
            )

        return ft.AlertDialog(
            title=ft.Text("Need More Information"),
            content=ft.Container(
                content=ft.Column(
                    question_controls,
                    tight=True,
                    scroll=ft.ScrollMode.AUTO
                ),
                width=500,
                height=400
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._on_cancel()),
                ft.ElevatedButton(
                    "Submit",
                    on_click=lambda e: self.on_submit(self.selections),
                    disabled=len(self.selections) < len(self.clarifications)
                )
            ]
        )

    def _on_selection(self, question_idx: int, value: str):
        self.selections[question_idx] = value
```

#### 3.3 Chart Display Component
**File**: `src/tempo_app/ui/components/chart_display.py`

```python
import flet as ft
from pathlib import Path

class ChartDisplay:
    def __init__(self, chart_result):
        self.chart = chart_result

    def build(self) -> ft.Container:
        """Build chart display with controls."""
        return ft.Container(
            content=ft.Column([
                # Chart image
                ft.Image(
                    src=str(self.chart.png_path),
                    fit=ft.ImageFit.CONTAIN,
                    expand=True
                ),

                # Controls
                ft.Row([
                    ft.IconButton(
                        icon=ft.icons.DOWNLOAD,
                        tooltip="Download PNG",
                        on_click=lambda e: self._download(self.chart.png_path)
                    ),
                    ft.IconButton(
                        icon=ft.icons.DOWNLOAD_OUTLINED,
                        tooltip="Download SVG",
                        on_click=lambda e: self._download(self.chart.svg_path)
                    ),
                    ft.IconButton(
                        icon=ft.icons.CODE,
                        tooltip="View Generated Code",
                        on_click=lambda e: self._show_code_dialog()
                    ),
                    ft.IconButton(
                        icon=ft.icons.REFRESH,
                        tooltip="Regenerate from Script",
                        on_click=lambda e: self._regenerate()
                    )
                ], alignment=ft.MainAxisAlignment.CENTER),

                # Metadata
                ft.ExpansionTile(
                    title=ft.Text("Chart Parameters"),
                    controls=[
                        ft.Text(f"Chart Type: {self.chart.metadata['chart_type']}"),
                        ft.Text(f"Variable(s): {', '.join(self.chart.metadata.get('variables', []))}"),
                        ft.Text(f"Site(s): {', '.join(self.chart.metadata.get('sites', []))}"),
                        ft.Text(f"Created: {self.chart.metadata['created_at']}")
                    ]
                )
            ]),
            border_radius=8,
            padding=10
        )

    def _show_code_dialog(self):
        """Show generated Python code in dialog."""
        # Implementation...
        pass
```

### Phase 4: Gemini API Integration

#### 4.1 Configuration
Add to `src/tempo_app/config.py`:

```python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Existing settings...

    # Gemini API settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_timeout: int = 30  # seconds
    enable_nl_charts: bool = True
    gemini_cache_ttl: int = 3600  # 1 hour
    gemini_max_queries_per_hour: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

Create `.env.example`:
```bash
# Gemini API Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
ENABLE_NL_CHARTS=true
```

#### 4.2 Rate Limiting & Caching
**File**: `src/tempo_app/integrations/rate_limiter.py`

```python
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        """
        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def can_make_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)

        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        return len(self.requests) < self.max_requests

    def record_request(self):
        """Record a new request."""
        self.requests.append(datetime.now())

    def time_until_available(self) -> Optional[int]:
        """Get seconds until next request is allowed."""
        if self.can_make_request():
            return 0

        oldest_request = self.requests[0]
        cutoff = oldest_request + timedelta(seconds=self.time_window)
        return (cutoff - datetime.now()).total_seconds()
```

Add to `GeminiQueryInterpreter`:
```python
class GeminiQueryInterpreter:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.cache = {}
        self.rate_limiter = RateLimiter(
            max_requests=100,
            time_window=3600  # 100 requests per hour
        )

    def interpret_query(self, query: str, context: Dict) -> ChartParameters:
        # Check rate limit
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.time_until_available()
            raise RateLimitError(
                f"Rate limit exceeded. Try again in {wait_time:.0f} seconds."
            )

        # Check cache first...
        cache_key = f"{query}:{hash(json.dumps(context, sort_keys=True))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Make API call
        self.rate_limiter.record_request()
        # ... rest of implementation
```

### Phase 5: Testing & Validation

#### 5.1 Unit Tests
**File**: `tests/test_chart_generator.py`

```python
import pytest
from src.tempo_app.core.chart_generator import ChartGenerator
import xarray as xr
import numpy as np

@pytest.fixture
def sample_data():
    """Create sample NO2 data for testing."""
    times = pd.date_range('2024-08-01', '2024-08-31', freq='H')
    lats = np.linspace(40.0, 41.0, 10)
    lons = np.linspace(-112.0, -111.0, 10)

    data = np.random.rand(len(times), len(lats), len(lons)) * 1e15

    return xr.Dataset({
        'NO2': (['time', 'lat', 'lon'], data)
    }, coords={
        'time': times,
        'lat': lats,
        'lon': lons
    })

def test_time_series_generation(sample_data, tmp_path):
    """Test time series chart generation."""
    generator = ChartGenerator(output_dir=tmp_path)

    result = generator.generate_time_series(
        data=sample_data,
        variable='NO2',
        sites=[Site(id='BV', name='Bountiful', lat=40.5, lon=-111.5)],
        title='Test Chart'
    )

    assert result.png_path.exists()
    assert result.svg_path.exists()
    assert result.script_path.exists()
    assert 'plt.plot' in result.script_code

def test_script_regeneration(sample_data, tmp_path):
    """Test that generated scripts can reproduce charts."""
    generator = ChartGenerator(output_dir=tmp_path)

    result = generator.generate_time_series(...)

    # Execute generated script
    exec(result.script_code)

    # Verify regenerated chart exists
    assert Path('chart_regenerated.png').exists()
```

**File**: `tests/test_gemini_integration.py`

```python
import pytest
from src.tempo_app.integrations.gemini_client import GeminiQueryInterpreter
from unittest.mock import Mock, patch

@pytest.fixture
def mock_gemini():
    with patch('google.generativeai.GenerativeModel') as mock:
        yield mock

def test_query_interpretation(mock_gemini):
    """Test NL query interpretation."""
    # Mock Gemini response
    mock_response = Mock()
    mock_response.text = json.dumps({
        'chart_type': 'time_series',
        'variables': ['NO2'],
        'sites': [{'site_id': 'BV', 'name': 'Bountiful'}],
        # ... rest of schema
    })
    mock_gemini.return_value.generate_content.return_value = mock_response

    interpreter = GeminiQueryInterpreter(api_key='test_key')

    result = interpreter.interpret_query(
        query='Show NO2 at Bountiful',
        context={'sites': [{'id': 'BV', 'name': 'Bountiful'}]}
    )

    assert result.chart_type == 'time_series'
    assert result.variables == ['NO2']
    assert result.sites[0]['site_id'] == 'BV'

def test_clarification_needed(mock_gemini):
    """Test ambiguous query handling."""
    mock_response = Mock()
    mock_response.text = json.dumps({
        'chart_type': 'time_series',
        'clarifications_needed': [{
            'question': 'Which site?',
            'options': ['BV', 'HC'],
            'context': 'Multiple sites found'
        }]
    })
    mock_gemini.return_value.generate_content.return_value = mock_response

    interpreter = GeminiQueryInterpreter(api_key='test_key')
    result = interpreter.interpret_query('Show NO2', context={})

    assert len(result.clarifications_needed) == 1
    assert 'Which site?' in result.clarifications_needed[0]['question']
```

#### 5.2 Integration Tests
**File**: `tests/test_nl_to_chart_flow.py`

```python
import pytest
from src.tempo_app.ui.pages.charts import ChartsPage

@pytest.mark.integration
async def test_full_nl_to_chart_flow(db_session, sample_dataset):
    """Test complete flow from NL query to chart generation."""
    page = ChartsPage(page=Mock(), db_session=db_session)

    # Simulate user query
    query = "Show NO2 trends at BV site for August 2024"

    # Interpret query
    params = await page._async_interpret(query, context=...)

    # Validate
    validation = page.validator.validate(params)
    assert validation.is_valid

    # Generate chart
    result = await page._async_generate_chart(params)

    # Verify outputs
    assert result.png_path.exists()
    assert result.script_path.exists()
    assert 'NO2' in result.metadata['variables']

@pytest.mark.integration
async def test_clarification_flow(db_session):
    """Test interactive clarification flow."""
    page = ChartsPage(page=Mock(), db_session=db_session)

    # Ambiguous query
    query = "Show trends"  # No variable or site specified

    params = await page._async_interpret(query, context=...)

    # Should request clarifications
    assert len(params.clarifications_needed) > 0

    # Simulate user providing clarifications
    clarifications = {
        0: "NO2",  # Variable
        1: "BV"    # Site
    }

    # Re-interpret with clarifications
    # ...
```

#### 5.3 Manual Testing Checklist

1. **Unambiguous Query Test**
   - Input: "Show NO₂ time series at Bountiful for August 2024"
   - Expected: Direct chart generation without clarifications
   - Verify: Chart shows NO₂ on Y-axis, dates on X-axis, data only for August

2. **Ambiguous Query Test**
   - Input: "Show trends"
   - Expected: Clarification dialog asking for variable and site
   - Verify: User can select from available options, chart generates after selection

3. **Invalid Query Test**
   - Input: "Show CO₂ at Mars"
   - Expected: Error message explaining CO₂ is not available
   - Verify: Helpful error with suggestions for valid variables

4. **Complex Query Test**
   - Input: "Compare weekday vs weekend NO₂ at BV when cloud fraction < 0.3"
   - Expected: Chart with two lines (weekday, weekend), only low-cloud data
   - Verify: Correct filtering applied

5. **Script Regeneration Test**
   - Generate chart from NL query
   - Navigate to saved chart folder
   - Run `python generate_chart.py`
   - Verify: `chart_regenerated.png` matches original

6. **API Error Handling**
   - Disconnect from internet
   - Try to generate chart
   - Expected: Clear error message, option to use manual chart builder
   - Verify: App doesn't crash, user can still work offline

7. **Rate Limit Test**
   - Make 101 requests in quick succession
   - Expected: 101st request shows "Rate limit exceeded" with countdown
   - Verify: User can still browse history, view saved charts

## Critical Files Summary

### New Files
| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/tempo_app/core/chart_generator.py` | Matplotlib chart generation | 500 |
| `src/tempo_app/integrations/gemini_client.py` | Gemini API client | 300 |
| `src/tempo_app/core/chart_validator.py` | Parameter validation | 200 |
| `src/tempo_app/integrations/rate_limiter.py` | API rate limiting | 100 |
| `src/tempo_app/ui/pages/charts.py` | Charts UI page | 400 |
| `src/tempo_app/ui/components/chart_display.py` | Chart viewer | 200 |
| `src/tempo_app/ui/components/clarification_dialog.py` | Interactive Q&A | 150 |
| `tests/test_chart_generator.py` | Chart generation tests | 300 |
| `tests/test_gemini_integration.py` | Gemini client tests | 250 |
| `tests/test_nl_to_chart_flow.py` | Integration tests | 200 |

### Modified Files
| File | Changes |
|------|---------|
| `src/tempo_app/storage/models.py` | Add `GeneratedChart` model |
| `src/tempo_app/ui/app.py` | Add charts page route |
| `src/tempo_app/config.py` | Add Gemini settings |
| `requirements.txt` | Add `google-generativeai` |

### Configuration Files
| File | Purpose |
|------|---------|
| `.env.example` | Gemini API key template |
| `alembic/versions/xxx_add_charts.py` | Database migration |

## Dependencies to Add

```txt
# requirements.txt additions
google-generativeai>=0.3.0  # Gemini API client
```

## Risks & Mitigation Strategies

### Risk 1: API Costs Exceed Free Tier
**Likelihood**: Medium
**Impact**: Medium

**Mitigation**:
- Implement aggressive caching (1 hour TTL for identical queries)
- Rate limiting (100 queries/hour per user)
- Usage monitoring dashboard
- Alert when approaching 80% of quota
- Fallback to manual chart builder when limit exceeded

**Monitoring**:
```python
class UsageMonitor:
    def track_api_call(self, user_id, cost):
        # Log to database
        # Check daily/monthly limits
        # Send alerts if needed
        pass
```

### Risk 2: Gemini Generates Invalid/Unsafe Parameters
**Likelihood**: Low
**Impact**: High

**Mitigation**:
- Strict JSON schema validation
- Database validation (dataset/site IDs must exist)
- Sandboxed script execution
- Never execute arbitrary code from Gemini
- Whitelist allowed chart types and parameters

**Example**:
```python
# SAFE: Gemini only returns data parameters
{
  "chart_type": "time_series",  # From whitelist
  "dataset_id": "uuid",  # Validated against DB
  "sites": ["BV"]  # Validated against DB
}

# UNSAFE: Don't allow this
{
  "custom_code": "import os; os.system('rm -rf /')"  # NEVER execute
}
```

### Risk 3: Slow API Response Times
**Likelihood**: Medium
**Impact**: Medium

**Mitigation**:
- Show loading state with progress indicator
- Timeout after 30 seconds
- Async UI (page remains responsive)
- Cache common queries
- Option to cancel request

**UX**:
```python
async def on_generate_click(self, e):
    self.loading.visible = True
    self.status_text.value = "Interpreting your request... (usually <5s)"

    try:
        with timeout(30):
            params = await self.gemini.interpret_query(...)
    except TimeoutError:
        self.status_text.value = "Request timed out. Try simplifying your query or use manual mode."
```

### Risk 4: Model Updates Break Reproducibility
**Likelihood**: Medium
**Impact**: Low

**Mitigation**:
- Generated Matplotlib scripts are source of truth
- Save Gemini model version in metadata
- Scripts work indefinitely without AI
- Pin to specific Gemini model version
- Document model used for each chart

**Metadata**:
```json
{
  "gemini_model": "gemini-2.0-flash-exp",
  "gemini_version": "2.0",
  "generated_at": "2024-08-15T14:32:00Z",
  "can_regenerate_without_ai": true
}
```

### Risk 5: Users Prefer Manual UI
**Likelihood**: Medium
**Impact**: Low

**Mitigation**:
- Implement both NL and manual interfaces
- A/B test to see which is more popular
- User preference setting
- Make NL optional (feature flag)

**Configuration**:
```python
class UserPreferences:
    default_chart_mode: Literal['nl', 'manual'] = 'nl'
    show_both_options: bool = True
```

## Alternative Approaches Considered

### Alternative 1: Pure Matplotlib UI (No Gemini)
**Pros**:
- Zero API costs
- Instant response
- Fully offline
- No external dependencies

**Cons**:
- Steeper learning curve
- More clicks for complex queries
- Less discoverable

**Decision**: Implement BOTH as described in plan

### Alternative 2: Gemini Generates Charts Directly
**Pros**:
- Simpler architecture
- Less code to maintain

**Cons**:
- ❌ Not reproducible (AI output varies)
- ❌ Can't control chart quality/formatting
- ❌ Sends atmospheric data to external API
- ❌ Can't tweak parameters precisely
- ❌ Unsuitable for publications

**Decision**: REJECTED - Fails scientific requirements

### Alternative 3: Local LLM (Llama, Mistral)
**Pros**:
- No API costs
- Full data privacy
- No rate limits
- Works offline

**Cons**:
- Requires GPU or slow CPU inference
- Larger application size
- Complex setup for end users
- Lower quality interpretations

**Decision**: DEFERRED - Consider for future if Gemini costs become prohibitive

### Alternative 4: Template-Based Query Builder
**Pros**:
- No AI needed
- Instant response
- Predictable results

**Cons**:
- Less flexible than NL
- Still requires learning templates
- Not as user-friendly

**Example**:
```
Template: "Show {variable} at {site} from {start} to {end} for {days} when {filters}"
Filled: "Show NO2 at BV from 2024-08-01 to 2024-08-31 for weekdays when cloud < 0.3"
```

**Decision**: Could combine with NL - templates help users learn query syntax

## Phased Rollout Recommendation

### Phase 1: MVP (Week 1-2)
**Scope**:
- Time series charts only
- Basic Gemini integration
- Simple validation
- Manual chart builder

**Goal**: Validate concept, get user feedback

### Phase 2: Full Feature Set (Week 3-4)
**Scope**:
- All 5 chart types
- Comprehensive clarification dialogs
- Chart history
- Script regeneration

**Goal**: Production-ready feature

### Phase 3: Polish & Optimization (Week 5+)
**Scope**:
- Advanced caching strategies
- Usage analytics
- Performance optimization
- Publication templates

**Goal**: Scale to many users

### Future Enhancements (Beyond Initial Release)
- AI-generated insights ("Anomaly detected on Aug 15")
- Batch chart generation
- Custom chart templates
- Multi-panel figures
- Export to PowerPoint/LaTeX
- Collaborative chart sharing

## Final Recommendation

**YES - Implement this hybrid NL-to-Matplotlib approach** with these critical success factors:

1. ✅ **Gemini as interpreter, not generator** - Preserves scientific control
2. ✅ **Save generated Matplotlib code** - Ensures reproducibility for publications
3. ✅ **Data privacy by design** - Only metadata sent to API
4. ✅ **Manual fallback** - Works when API is down
5. ✅ **Aggressive caching** - Minimizes costs
6. ✅ **Strict validation** - Prevents invalid charts

This architecture delivers the user-friendly NL interface while maintaining the scientific rigor required for research publications. The saved Matplotlib scripts ensure that charts can be regenerated identically for years to come, regardless of AI model changes.

The key insight: **Use AI for the interface, not the implementation**. Gemini translates human intent into structured parameters, but Matplotlib does the actual work. This separation preserves reproducibility while adding convenience.
