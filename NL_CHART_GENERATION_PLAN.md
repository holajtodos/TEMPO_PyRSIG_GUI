# Natural Language Chart Generation for TEMPO Analyzer - Implementation Plan

## Executive Summary

Implement a **Natural Language Chart Generation** feature using **Google Gemini API** (direct, no PandasAI library). This feature will:

1. Accept natural language queries from users (e.g., "Plot NO₂ trends at BV site by hour")
2. Generate executable matplotlib code via Gemini
3. Execute the code safely with user's DataFrame
4. **Save both the code and the plot** for reproducibility
5. Allow users to **edit the generated code** and re-run it

**Key Principle:** Every plot must be reproducible - code is saved alongside the plot and can be modified.

---

## Architecture Overview

```
User Query: "Show NO₂ trends at BV during weekdays"
    ↓
1. Load Dataset → Prepare DataFrame with schema
    ↓
2. Send Query + DataFrame Schema → Gemini API
    ↓
3. Gemini Returns: Python Code (matplotlib + pandas)
    ↓
4. Execute Code in Safe Sandbox
    ↓
5. Capture: Plot PNG + Generated Code
    ↓
6. Save to Database: Analysis(query, code, plot_path, dataset_id)
    ↓
7. Display in UI: Code Editor (editable) + Plot Viewer
    ↓
8. User can Edit Code → Re-run → Update Analysis Record
```

---

## Why Direct Gemini API (Not PandasAI)?

### Advantages of This Approach
- **Full Control**: We control exactly what code gets generated and executed
- **Reproducibility**: Every plot has its source code saved
- **Editability**: Users can modify generated code for fine-tuning
- **Lightweight**: No heavy dependencies (just `google-generativeai`)
- **Transparency**: Users see exactly what code runs on their data
- **Cost Efficient**: Direct API calls, no middleware overhead

### Trade-offs & Mitigations
- **Code Execution Risk**: Executing LLM-generated code could be dangerous
    - *Mitigation*: Sandboxed execution with restricted globals (no file I/O, no subprocess, only safe modules)
- **Code Quality**: Gemini might generate suboptimal or incorrect code
    - *Mitigation*: Users can edit and fix the code; save corrected versions
- **Data Privacy**: DataFrame schema sent to Google
    - *Mitigation*: Only column names/types sent, not actual data values

---

## Implementation Details

### 1. Dependencies

**Add to `requirements.txt`:**
```txt
google-generativeai>=0.8.0  # Gemini API client
matplotlib>=3.8.0           # Already installed
pandas>=2.1.0               # Already installed
numpy>=1.24.0               # Already installed
```

**Installation:**
```bash
pip install google-generativeai
```

---

### 2. Database Model - Analysis Storage

**File: `src/tempo_app/storage/models.py`**

Add a new dataclass to store analysis results:

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

@dataclass
class Analysis:
    """
    Stores a saved chart analysis with its code.

    Attributes:
        id: Unique identifier
        dataset_id: Which dataset this analysis uses
        name: User-friendly name (e.g., "NO₂ Hourly Trends")
        query: Original natural language query
        code: Generated matplotlib code (editable)
        plot_path: Path to saved PNG file
        created_at: When first generated
        updated_at: When code was last edited/re-run
        error_message: If execution failed, store error here
    """
    id: UUID
    dataset_id: UUID
    name: str
    query: str
    code: str
    plot_path: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None

    @staticmethod
    def new(dataset_id: UUID, query: str, code: str, plot_path: str, name: str = "") -> "Analysis":
        """Create a new analysis record."""
        now = datetime.now()
        return Analysis(
            id=uuid4(),
            dataset_id=dataset_id,
            name=name or f"Analysis {now.strftime('%Y-%m-%d %H:%M')}",
            query=query,
            code=code,
            plot_path=plot_path,
            created_at=now,
            updated_at=now,
            error_message=None
        )
```

**Database Schema (SQLite):**

```sql
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    name TEXT NOT NULL,
    query TEXT NOT NULL,
    code TEXT NOT NULL,
    plot_path TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    error_message TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE INDEX idx_analyses_dataset ON analyses(dataset_id);
CREATE INDEX idx_analyses_created ON analyses(created_at DESC);
```

**Add to `src/tempo_app/storage/database.py`:**

```python
class DatabaseManager:
    # ... existing methods ...

    def save_analysis(self, analysis: Analysis) -> None:
        """Save or update an analysis record."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analyses
                (id, dataset_id, name, query, code, plot_path, created_at, updated_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(analysis.id),
                str(analysis.dataset_id),
                analysis.name,
                analysis.query,
                analysis.code,
                analysis.plot_path,
                analysis.created_at,
                analysis.updated_at,
                analysis.error_message
            ))
            conn.commit()

    def get_analyses_for_dataset(self, dataset_id: UUID) -> list[Analysis]:
        """Retrieve all analyses for a dataset, newest first."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, dataset_id, name, query, code, plot_path,
                       created_at, updated_at, error_message
                FROM analyses
                WHERE dataset_id = ?
                ORDER BY created_at DESC
            """, (str(dataset_id),)).fetchall()

            return [Analysis(
                id=UUID(row[0]),
                dataset_id=UUID(row[1]),
                name=row[2],
                query=row[3],
                code=row[4],
                plot_path=row[5],
                created_at=row[6],
                updated_at=row[7],
                error_message=row[8]
            ) for row in rows]

    def get_analysis(self, analysis_id: UUID) -> Analysis | None:
        """Retrieve a specific analysis by ID."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT id, dataset_id, name, query, code, plot_path,
                       created_at, updated_at, error_message
                FROM analyses
                WHERE id = ?
            """, (str(analysis_id),)).fetchone()

            if not row:
                return None

            return Analysis(
                id=UUID(row[0]),
                dataset_id=UUID(row[1]),
                name=row[2],
                query=row[3],
                code=row[4],
                plot_path=row[5],
                created_at=row[6],
                updated_at=row[7],
                error_message=row[8]
            )

    def delete_analysis(self, analysis_id: UUID) -> None:
        """Delete an analysis and its plot file."""
        analysis = self.get_analysis(analysis_id)
        if analysis:
            # Delete plot file
            plot_path = Path(analysis.plot_path)
            if plot_path.exists():
                plot_path.unlink()

            # Delete DB record
            with self._get_connection() as conn:
                conn.execute("DELETE FROM analyses WHERE id = ?", (str(analysis_id),))
                conn.commit()
```

---

### 3. Configuration - Gemini API Key

**File: `src/tempo_app/core/config.py`**

Add Gemini API key to the default config:

```python
DEFAULT_CONFIG = {
    "data_dir": None,              # Custom data storage location
    "font_scale": 1.0,             # UI font scaling (0.8-1.5)
    "theme_mode": "light",         # Reserved for future
    "download_workers": 8,         # Parallel download threads
    "rsig_api_key": "",            # NASA RSIG API key (optional)
    "gemini_api_key": "",          # Google Gemini API key for AI analysis
}
```

No other changes needed - the existing `ConfigManager` class handles get/set automatically.

---

### 4. Core Service - Chart Code Generator

**File: `src/tempo_app/core/chart_generator.py` (NEW FILE)**

This is the heart of the system - handles Gemini API communication and code execution.

```python
"""
Chart code generator using Google Gemini API.

This module generates matplotlib code from natural language queries
and executes it safely in a sandboxed environment.
"""

import io
import traceback
from pathlib import Path
from typing import Any

import google.generativeai as genai
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tempo_app.core.config import ConfigManager

# Use non-interactive backend for server-side rendering
matplotlib.use('Agg')


class ChartGenerationError(Exception):
    """Raised when chart code generation or execution fails."""
    pass


class ChartGenerator:
    """
    Generates and executes matplotlib code using Google Gemini API.

    Usage:
        generator = ChartGenerator()
        code = generator.generate_code(
            query="Plot NO2 trends by hour",
            df_schema={"columns": [...], "dtypes": {...}}
        )
        plot_path = generator.execute_code(code, df)
    """

    def __init__(self):
        """Initialize the chart generator with Gemini API."""
        self.config = ConfigManager()
        self._setup_gemini()

    def _setup_gemini(self) -> None:
        """Configure Gemini API with user's key."""
        api_key = self.config.get("gemini_api_key")

        if not api_key:
            raise ChartGenerationError(
                "Gemini API key not configured. "
                "Please add your API key in Settings."
            )

        genai.configure(api_key=api_key)

        # Use gemini-1.5-flash for speed and cost efficiency
        # Alternative: gemini-1.5-pro for better quality
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_code(
        self,
        query: str,
        df_schema: dict[str, Any]
    ) -> str:
        """
        Generate matplotlib code from a natural language query.

        Args:
            query: Natural language description (e.g., "Plot NO2 by hour")
            df_schema: DataFrame structure info
                {
                    "columns": ["TIME", "NO2_TropVCD", "site_code", ...],
                    "dtypes": {"TIME": "datetime64[ns]", "NO2_TropVCD": "float64", ...},
                    "shape": (1000, 8),
                    "sample_values": {"site_code": ["BV", "LA", "BA"], ...}
                }

        Returns:
            Python code string that generates a matplotlib chart

        Raises:
            ChartGenerationError: If API call fails or response is invalid
        """

        # Build the prompt for Gemini
        prompt = self._build_prompt(query, df_schema)

        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)

            if not response.text:
                raise ChartGenerationError("Gemini returned empty response")

            # Extract code from response (remove markdown formatting if present)
            code = self._extract_code(response.text)

            return code

        except Exception as e:
            raise ChartGenerationError(f"Failed to generate code: {str(e)}")

    def _build_prompt(self, query: str, df_schema: dict[str, Any]) -> str:
        """
        Construct the prompt for Gemini.

        This is the critical part - a well-crafted prompt ensures good code.
        """

        columns_str = ", ".join(df_schema["columns"])
        dtypes_str = "\n".join([f"  - {col}: {dtype}"
                                for col, dtype in df_schema["dtypes"].items()])

        # Include sample values for categorical columns
        samples_str = ""
        if "sample_values" in df_schema:
            samples_str = "\nSample categorical values:\n" + "\n".join([
                f"  - {col}: {values}"
                for col, values in df_schema["sample_values"].items()
            ])

        prompt = f"""You are a data visualization expert. Generate Python code using matplotlib and pandas to create a chart.

DATASET INFORMATION:
- Available DataFrame: `df` (already loaded in memory)
- Columns: {columns_str}
- Data types:
{dtypes_str}
- Shape: {df_schema.get('shape', 'Unknown')} rows
{samples_str}

USER REQUEST:
{query}

REQUIREMENTS:
1. Use ONLY these imports (already available):
   - pandas as pd
   - numpy as np
   - matplotlib.pyplot as plt

2. The DataFrame `df` is already loaded - DO NOT load data

3. Generate a single matplotlib chart

4. MUST save the figure using:
   plt.savefig('OUTPUT_PATH', dpi=150, bbox_inches='tight')

5. Include proper:
   - Title (clear and descriptive)
   - Axis labels with units
   - Legend (if multiple series)
   - Grid (if appropriate)

6. Use appropriate chart type:
   - Time series → Line plot
   - Comparisons → Bar chart
   - Distributions → Histogram or box plot
   - Correlations → Scatter plot

7. Handle missing data appropriately (dropna or fillna)

8. Set figure size for readability: plt.figure(figsize=(10, 6))

9. DO NOT use plt.show() - only plt.savefig()

10. Return ONLY executable Python code - no explanations, no markdown

EXAMPLE OUTPUT FORMAT:
import matplotlib.pyplot as plt
import pandas as pd

# Filter data for specific site
df_filtered = df[df['site_code'] == 'BV']

# Group by hour and calculate mean
hourly_mean = df_filtered.groupby(df_filtered['TIME'].dt.hour)['NO2_TropVCD'].mean()

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2)
plt.xlabel('Hour of Day')
plt.ylabel('NO₂ Tropospheric VCD (molecules/cm²)')
plt.title('Average NO₂ Levels by Hour at BV Site')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('OUTPUT_PATH', dpi=150, bbox_inches='tight')
plt.close()

Now generate code for the user's request above.
"""
        return prompt

    def _extract_code(self, response_text: str) -> str:
        """
        Extract Python code from Gemini's response.

        Gemini might wrap code in markdown:
        ```python
        code here
        ```

        This function strips the markdown and returns clean code.
        """

        # Remove markdown code blocks
        if "```python" in response_text:
            # Extract content between ```python and ```
            start = response_text.find("```python") + len("```python")
            end = response_text.find("```", start)
            code = response_text[start:end].strip()
        elif "```" in response_text:
            # Generic code block
            start = response_text.find("```") + len("```")
            end = response_text.find("```", start)
            code = response_text[start:end].strip()
        else:
            # No markdown, use as-is
            code = response_text.strip()

        return code

    def execute_code(
        self,
        code: str,
        df: pd.DataFrame,
        output_path: Path
    ) -> Path:
        """
        Execute generated code in a safe sandbox.

        Args:
            code: Python code to execute
            df: DataFrame to make available to the code
            output_path: Where to save the plot

        Returns:
            Path to the generated plot

        Raises:
            ChartGenerationError: If execution fails
        """

        # Replace OUTPUT_PATH placeholder with actual path
        code = code.replace("'OUTPUT_PATH'", f"'{output_path}'")
        code = code.replace('"OUTPUT_PATH"', f'"{output_path}"')

        # Create safe execution environment
        safe_globals = {
            # Data libraries
            'pd': pd,
            'np': np,
            'plt': plt,

            # User's data
            'df': df,

            # Disable dangerous builtins
            '__builtins__': {
                # Allow only safe built-ins
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'print': print,  # Allow for debugging
            }
        }

        try:
            # Execute the code
            exec(code, safe_globals)

            # Verify plot was created
            if not output_path.exists():
                raise ChartGenerationError(
                    "Code executed but no plot file was created. "
                    "Make sure code includes plt.savefig()"
                )

            return output_path

        except Exception as e:
            # Capture full traceback for debugging
            error_details = traceback.format_exc()
            raise ChartGenerationError(
                f"Code execution failed:\n{error_details}"
            )

    def get_dataframe_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Extract schema information from a DataFrame.

        This is sent to Gemini so it understands the data structure.

        Args:
            df: The DataFrame to analyze

        Returns:
            Schema dictionary with columns, types, and sample values
        """

        schema = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample_values": {}
        }

        # For categorical columns, provide sample unique values
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 20:  # Only if reasonable number
                    schema["sample_values"][col] = unique_vals.tolist()[:10]

        return schema
```

---

### 5. Data Preparation - DataFrame Converter

**File: `src/tempo_app/core/df_converter.py` (NEW FILE)**

Convert xarray Datasets to pandas DataFrames optimized for AI analysis.

```python
"""
Converts TEMPO xarray Datasets to pandas DataFrames for AI analysis.

Flattens multi-dimensional data (TIME, LAT, LON) into a tabular format.
"""

import pandas as pd
import xarray as xr
from pathlib import Path


class DataFrameConverter:
    """
    Converts TEMPO NetCDF datasets to pandas DataFrames.

    The conversion flattens the 3D structure (TIME, LAT, LON) into rows,
    making it suitable for pandas/matplotlib operations.
    """

    @staticmethod
    def dataset_to_dataframe(
        dataset_path: Path,
        include_coords: bool = True,
        downsample: int | None = None
    ) -> pd.DataFrame:
        """
        Convert a TEMPO dataset to a pandas DataFrame.

        Args:
            dataset_path: Path to the NetCDF file
            include_coords: Whether to include LAT/LON columns
            downsample: If set, take every Nth point to reduce size

        Returns:
            DataFrame with columns:
                - TIME (datetime)
                - NO2_TropVCD (float)
                - HCHO_TotVCD (float)
                - FNR (float, if available)
                - LAT (float, if include_coords=True)
                - LON (float, if include_coords=True)

        Example:
            df = DataFrameConverter.dataset_to_dataframe(
                Path("dataset.nc"),
                downsample=10  # Take every 10th point
            )
        """

        # Load xarray dataset
        ds = xr.open_dataset(dataset_path)

        # Convert to DataFrame
        df = ds.to_dataframe().reset_index()

        # Apply downsampling if requested
        if downsample and downsample > 1:
            df = df.iloc[::downsample]

        # Remove coordinate columns if not needed (saves memory)
        if not include_coords:
            df = df.drop(columns=['LAT', 'LON'], errors='ignore')

        # Ensure TIME is datetime
        if 'TIME' in df.columns:
            df['TIME'] = pd.to_datetime(df['TIME'])

        # Sort by time for better plotting
        if 'TIME' in df.columns:
            df = df.sort_values('TIME').reset_index(drop=True)

        return df

    @staticmethod
    def add_site_data(
        df: pd.DataFrame,
        sites: list[tuple[str, str, float, float]],
        tolerance: float = 0.01
    ) -> pd.DataFrame:
        """
        Add site_code and site_name columns by matching LAT/LON.

        Args:
            df: DataFrame with LAT/LON columns
            sites: List of (code, name, lat, lon) tuples
            tolerance: Lat/lon matching tolerance in degrees

        Returns:
            DataFrame with added 'site_code' and 'site_name' columns

        Example:
            sites = [
                ("BV", "Bakersfield - Planz", 35.3528, -119.0369),
                ("LA", "Los Angeles", 34.0522, -118.2437),
            ]
            df = DataFrameConverter.add_site_data(df, sites)
        """

        if 'LAT' not in df.columns or 'LON' not in df.columns:
            raise ValueError("DataFrame must have LAT and LON columns")

        # Initialize columns
        df['site_code'] = None
        df['site_name'] = None

        # Match each row to nearest site
        for code, name, site_lat, site_lon in sites:
            # Find points within tolerance
            mask = (
                (abs(df['LAT'] - site_lat) < tolerance) &
                (abs(df['LON'] - site_lon) < tolerance)
            )

            df.loc[mask, 'site_code'] = code
            df.loc[mask, 'site_name'] = name

        return df

    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful time-based columns for analysis.

        Args:
            df: DataFrame with TIME column

        Returns:
            DataFrame with added columns:
                - hour: Hour of day (0-23)
                - day_of_week: Day name (Monday, Tuesday, ...)
                - is_weekend: Boolean
                - date: Date only (no time)

        Example:
            df = DataFrameConverter.add_temporal_features(df)
            # Now can query: "Plot NO2 on weekdays"
        """

        if 'TIME' not in df.columns:
            raise ValueError("DataFrame must have TIME column")

        df['hour'] = df['TIME'].dt.hour
        df['day_of_week'] = df['TIME'].dt.day_name()
        df['is_weekend'] = df['TIME'].dt.dayofweek >= 5
        df['date'] = df['TIME'].dt.date

        return df
```

---

### 6. UI Page - AI Analysis Interface

**File: `src/tempo_app/ui/pages/ai_analysis.py` (NEW FILE)**

This is the main UI page where users interact with the AI chart generator.

```python
"""
AI Analysis page - Natural language chart generation interface.

Users can:
1. Select a dataset
2. Type natural language queries
3. View generated code (and edit it)
4. See the resulting plot
5. Save and manage analyses
"""

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import UUID

import flet as ft
import pandas as pd

from tempo_app.core.chart_generator import ChartGenerator, ChartGenerationError
from tempo_app.core.df_converter import DataFrameConverter
from tempo_app.storage.database import DatabaseManager
from tempo_app.storage.models import Analysis, Dataset
from tempo_app.ui.theme import (
    Colors,
    Spacing,
    Typography,
    card_style,
    section_header_style,
    primary_button_style
)


class AIAnalysisPage:
    """
    AI-powered chart generation page.

    Layout:
    ┌─────────────────────────────────────────────┐
    │ Dataset: [Select Dataset ▾]                │
    ├─────────────────────────────────────────────┤
    │ Query: [Type your question here...       ] │
    │        [Generate Chart]                     │
    ├─────────────────────────────────────────────┤
    │ Generated Code:              [Edit] [Run]   │
    │ ┌─────────────────────────────────────────┐ │
    │ │ import matplotlib.pyplot as plt         │ │
    │ │ ...                                     │ │
    │ └─────────────────────────────────────────┘ │
    ├─────────────────────────────────────────────┤
    │ Result:                                     │
    │ [Chart displayed here]                      │
    ├─────────────────────────────────────────────┤
    │ Saved Analyses:                             │
    │ • NO₂ vs HCHO (2 min ago)                  │
    │ • Hourly trends (5 min ago)                │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.db = DatabaseManager()
        self.generator = ChartGenerator()

        # State
        self.selected_dataset: Dataset | None = None
        self.current_df: pd.DataFrame | None = None
        self.current_analysis: Analysis | None = None

        # UI Components (initialized in build())
        self.dataset_dropdown: ft.Dropdown = None
        self.query_field: ft.TextField = None
        self.code_editor: ft.TextField = None
        self.plot_image: ft.Image = None
        self.status_text: ft.Text = None
        self.history_list: ft.Column = None

    def build(self) -> ft.Control:
        """Build the page UI."""

        # Dataset selector
        self.dataset_dropdown = ft.Dropdown(
            label="Select Dataset",
            hint_text="Choose a dataset to analyze",
            on_change=self._on_dataset_selected,
            expand=True
        )

        # Load available datasets
        self._refresh_datasets()

        # Query input
        self.query_field = ft.TextField(
            label="Natural Language Query",
            hint_text="e.g., Plot NO₂ trends by hour at BV site",
            multiline=False,
            expand=True,
            on_submit=self._on_generate_clicked
        )

        generate_btn = ft.ElevatedButton(
            "Generate Chart",
            icon=ft.icons.AUTO_AWESOME,
            on_click=self._on_generate_clicked,
            style=primary_button_style()
        )

        # Code editor
        self.code_editor = ft.TextField(
            label="Generated Code (Editable)",
            multiline=True,
            min_lines=10,
            max_lines=20,
            expand=True,
            text_style=ft.TextStyle(
                font_family="Consolas",
                size=Typography.BODY_SMALL
            ),
            read_only=False
        )

        edit_btn = ft.TextButton(
            "Run Code",
            icon=ft.icons.PLAY_ARROW,
            on_click=self._on_run_code_clicked
        )

        save_btn = ft.TextButton(
            "Save Analysis",
            icon=ft.icons.SAVE,
            on_click=self._on_save_clicked
        )

        # Plot viewer
        self.plot_image = ft.Image(
            visible=False,
            width=800,
            height=600,
            fit=ft.ImageFit.CONTAIN
        )

        # Status indicator
        self.status_text = ft.Text(
            "",
            size=Typography.BODY_SMALL,
            color=Colors.INFO,
            italic=True
        )

        # Saved analyses history
        self.history_list = ft.Column(
            spacing=Spacing.SM,
            scroll=ft.ScrollMode.AUTO,
            height=200
        )

        # Layout
        return ft.Container(
            content=ft.Column([
                # Header
                ft.Text(
                    "AI Chart Analysis",
                    **section_header_style()
                ),

                # Dataset selector
                ft.Container(
                    content=self.dataset_dropdown,
                    **card_style(),
                    padding=Spacing.MD
                ),

                # Query input
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            self.query_field,
                            generate_btn
                        ]),
                        self.status_text
                    ]),
                    **card_style(),
                    padding=Spacing.MD
                ),

                # Code editor
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text("Generated Code", weight=ft.FontWeight.BOLD),
                            ft.Row([edit_btn, save_btn], spacing=Spacing.SM)
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        self.code_editor
                    ]),
                    **card_style(),
                    padding=Spacing.MD
                ),

                # Plot viewer
                ft.Container(
                    content=ft.Column([
                        ft.Text("Result", weight=ft.FontWeight.BOLD),
                        self.plot_image
                    ]),
                    **card_style(),
                    padding=Spacing.MD
                ),

                # History
                ft.Container(
                    content=ft.Column([
                        ft.Text("Saved Analyses", weight=ft.FontWeight.BOLD),
                        self.history_list
                    ]),
                    **card_style(),
                    padding=Spacing.MD
                )
            ], spacing=Spacing.LG, scroll=ft.ScrollMode.AUTO),
            padding=Spacing.PAGE_HORIZONTAL
        )

    def _refresh_datasets(self):
        """Load datasets into dropdown."""
        datasets = self.db.get_all_datasets()
        self.dataset_dropdown.options = [
            ft.dropdown.Option(key=str(ds.id), text=ds.name)
            for ds in datasets
            if ds.is_complete  # Only show complete datasets
        ]
        self.page.update()

    async def _on_dataset_selected(self, e):
        """Handle dataset selection."""
        dataset_id = UUID(e.control.value)
        self.selected_dataset = self.db.get_dataset(dataset_id)

        if not self.selected_dataset:
            return

        # Load dataset into DataFrame
        self.status_text.value = "Loading dataset..."
        self.page.update()

        try:
            # Run in thread to avoid blocking UI
            self.current_df = await asyncio.to_thread(
                DataFrameConverter.dataset_to_dataframe,
                Path(self.selected_dataset.file_path),
                include_coords=True,
                downsample=None  # Load full data
            )

            # Add temporal features for easier querying
            self.current_df = DataFrameConverter.add_temporal_features(
                self.current_df
            )

            # Add site data if available
            sites = self.db.get_all_sites()
            if sites:
                site_data = [(s.code, s.name, s.latitude, s.longitude) for s in sites]
                self.current_df = DataFrameConverter.add_site_data(
                    self.current_df,
                    site_data
                )

            self.status_text.value = f"Loaded {len(self.current_df):,} rows"
            self.status_text.color = Colors.SUCCESS

            # Load analysis history
            self._refresh_history()

        except Exception as ex:
            self.status_text.value = f"Error loading dataset: {str(ex)}"
            self.status_text.color = Colors.ERROR

        self.page.update()

    async def _on_generate_clicked(self, e):
        """Handle 'Generate Chart' button click."""

        if not self.selected_dataset or self.current_df is None:
            self._show_error("Please select a dataset first")
            return

        query = self.query_field.value.strip()
        if not query:
            self._show_error("Please enter a query")
            return

        self.status_text.value = "Generating code with AI..."
        self.status_text.color = Colors.INFO
        self.page.update()

        try:
            # Get DataFrame schema
            schema = await asyncio.to_thread(
                self.generator.get_dataframe_schema,
                self.current_df
            )

            # Generate code via Gemini API
            code = await asyncio.to_thread(
                self.generator.generate_code,
                query,
                schema
            )

            # Display code
            self.code_editor.value = code
            self.status_text.value = "Code generated. Click 'Run Code' to execute."
            self.status_text.color = Colors.SUCCESS
            self.page.update()

            # Auto-execute
            await self._execute_code(code, query)

        except ChartGenerationError as ex:
            self._show_error(f"Generation failed: {str(ex)}")
        except Exception as ex:
            self._show_error(f"Unexpected error: {str(ex)}")

    async def _on_run_code_clicked(self, e):
        """Handle 'Run Code' button (for edited code)."""
        code = self.code_editor.value.strip()
        if not code:
            self._show_error("No code to execute")
            return

        query = self.query_field.value.strip() or "Custom plot"
        await self._execute_code(code, query)

    async def _execute_code(self, code: str, query: str):
        """Execute the code and display the plot."""

        if self.current_df is None:
            self._show_error("No dataset loaded")
            return

        self.status_text.value = "Executing code..."
        self.status_text.color = Colors.INFO
        self.page.update()

        try:
            # Create output path
            output_dir = Path.home() / ".tempo_analyzer" / "analyses"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"plot_{timestamp}.png"

            # Execute code
            plot_path = await asyncio.to_thread(
                self.generator.execute_code,
                code,
                self.current_df,
                output_path
            )

            # Display plot
            self.plot_image.src = str(plot_path)
            self.plot_image.visible = True

            self.status_text.value = "Chart generated successfully!"
            self.status_text.color = Colors.SUCCESS

            # Create analysis object (not saved yet)
            self.current_analysis = Analysis.new(
                dataset_id=self.selected_dataset.id,
                query=query,
                code=code,
                plot_path=str(plot_path)
            )

        except ChartGenerationError as ex:
            self._show_error(f"Execution failed: {str(ex)}")
        except Exception as ex:
            self._show_error(f"Unexpected error: {str(ex)}")

        self.page.update()

    async def _on_save_clicked(self, e):
        """Save the current analysis to database."""

        if not self.current_analysis:
            self._show_error("No analysis to save")
            return

        try:
            self.db.save_analysis(self.current_analysis)
            self.status_text.value = "Analysis saved!"
            self.status_text.color = Colors.SUCCESS

            # Refresh history
            self._refresh_history()

        except Exception as ex:
            self._show_error(f"Save failed: {str(ex)}")

        self.page.update()

    def _refresh_history(self):
        """Load saved analyses for current dataset."""

        if not self.selected_dataset:
            return

        analyses = self.db.get_analyses_for_dataset(self.selected_dataset.id)

        self.history_list.controls.clear()

        for analysis in analyses:
            # Calculate time ago
            time_ago = self._format_time_ago(analysis.created_at)

            item = ft.ListTile(
                title=ft.Text(analysis.name),
                subtitle=ft.Text(f"{analysis.query} • {time_ago}"),
                leading=ft.Icon(ft.icons.INSERT_CHART),
                on_click=lambda e, a=analysis: self._load_analysis(a)
            )
            self.history_list.controls.append(item)

        self.page.update()

    def _load_analysis(self, analysis: Analysis):
        """Load a saved analysis into the editor."""
        self.current_analysis = analysis
        self.query_field.value = analysis.query
        self.code_editor.value = analysis.code
        self.plot_image.src = analysis.plot_path
        self.plot_image.visible = True
        self.page.update()

    def _show_error(self, message: str):
        """Display error message."""
        self.status_text.value = message
        self.status_text.color = Colors.ERROR
        self.page.update()

    @staticmethod
    def _format_time_ago(dt: datetime) -> str:
        """Format datetime as 'X mins ago'."""
        delta = datetime.now() - dt

        if delta.seconds < 60:
            return "Just now"
        elif delta.seconds < 3600:
            mins = delta.seconds // 60
            return f"{mins} min{'s' if mins != 1 else ''} ago"
        elif delta.seconds < 86400:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            return dt.strftime("%Y-%m-%d")
```

---

### 7. Settings Page Integration

**File: `src/tempo_app/ui/pages/settings.py`**

Add Gemini API key configuration section (add to existing settings page):

```python
# Add this to the existing SettingsPage class

def _build_ai_section(self) -> ft.Container:
    """Build AI configuration section."""

    gemini_key_field = ft.TextField(
        label="Google Gemini API Key",
        hint_text="Enter your Gemini API key",
        password=True,
        can_reveal_password=True,
        value=self.config.get("gemini_api_key"),
        on_change=self._on_gemini_key_changed,
        expand=True
    )

    help_text = ft.Text(
        "Get a free API key at: https://makersuite.google.com/app/apikey",
        size=Typography.BODY_SMALL,
        color=Colors.INFO,
        italic=True
    )

    return ft.Container(
        content=ft.Column([
            ft.Text("AI Analysis", **section_header_style()),
            gemini_key_field,
            help_text,
            ft.Text(
                "⚠️ The API key is stored locally and used only for generating chart code.",
                size=Typography.BODY_SMALL,
                color=Colors.WARNING
            )
        ], spacing=Spacing.SM),
        **card_style(),
        padding=Spacing.MD
    )

def _on_gemini_key_changed(self, e):
    """Handle Gemini API key change."""
    self.config.set("gemini_api_key", e.control.value)
```

Then add this section to the settings page layout:
```python
# In the build() method, add:
self._build_ai_section(),
```

---

### 8. Navigation Integration

**File: `src/tempo_app/ui/shell.py`**

Add the new AI Analysis page to navigation:

```python
# Add import
from tempo_app.ui.pages.ai_analysis import AIAnalysisPage

# In the AppShell class __init__, add to page_cache:
self.page_cache = {
    "/library": None,
    "/new": None,
    "/batch": None,
    "/workspace": None,
    "/export": None,
    "/ai": None,  # NEW
    "/inspect": None,
    "/settings": None,
}

# In _build_nav_bar(), add tab:
ft.Tab(
    text="AI Analysis",
    icon=ft.icons.AUTO_AWESOME,
    content=ft.Container()  # Lazy loaded
),

# In _on_tab_changed(), add route:
routes = ["/library", "/new", "/batch", "/workspace", "/export", "/ai", "/inspect"]

# In _load_page(), add case:
case "/ai":
    if not self.page_cache["/ai"]:
        self.page_cache["/ai"] = AIAnalysisPage(self.page).build()
    return self.page_cache["/ai"]
```

---

## Security Considerations

### 1. Code Execution Sandbox

The `execute_code()` function restricts available globals:

```python
safe_globals = {
    'pd': pd,
    'np': np,
    'plt': plt,
    'df': df,
    '__builtins__': {
        # Only safe built-ins - NO file I/O, NO subprocess, NO imports
        'len': len, 'range': range, 'sum': sum, ...
    }
}
```

**What's blocked:**
- ❌ File operations (`open()`, `Path.write_text()`)
- ❌ Subprocess execution (`os.system()`, `subprocess.run()`)
- ❌ Dynamic imports (`__import__()`, `importlib`)
- ❌ Network access (`requests`, `urllib`)
- ❌ System modification (`os.remove()`, `shutil.rmtree()`)

**What's allowed:**
- ✅ Data manipulation (pandas, numpy)
- ✅ Plotting (matplotlib)
- ✅ Basic Python operations (math, string, list operations)

### 2. Data Privacy

**What's sent to Gemini:**
- ✅ Column names (`["TIME", "NO2_TropVCD", ...]`)
- ✅ Data types (`{"TIME": "datetime64[ns]", ...}`)
- ✅ Sample categorical values (`{"site_code": ["BV", "LA"]}`)

**What's NOT sent:**
- ❌ Actual data values
- ❌ Full DataFrame contents
- ❌ User's file paths

### 3. API Key Storage

- Stored in `~/.tempo_analyzer/config.json` (user's home directory)
- Not committed to version control (add to `.gitignore`)
- Password field in UI (not visible on screen)

---

## Error Handling

### Gemini API Errors

```python
try:
    code = generator.generate_code(query, schema)
except ChartGenerationError as e:
    if "API key" in str(e):
        # Show setup instructions
        show_error("Please configure your Gemini API key in Settings")
    elif "quota" in str(e).lower():
        # Rate limit hit
        show_error("API quota exceeded. Please try again later.")
    else:
        # Generic error
        show_error(f"Code generation failed: {e}")
```

### Code Execution Errors

```python
try:
    plot_path = generator.execute_code(code, df, output_path)
except ChartGenerationError as e:
    # Save error to analysis record
    analysis.error_message = str(e)
    db.save_analysis(analysis)

    # Show user-friendly error
    show_error("Code execution failed. Check the code for errors.")
```

### Dataset Loading Errors

```python
try:
    df = DataFrameConverter.dataset_to_dataframe(path)
except Exception as e:
    show_error(f"Failed to load dataset: {e}")
    # Suggest using a different dataset or re-downloading
```

---

## Testing Strategy

### 1. Unit Tests

**Test file: `tests/test_chart_generator.py`**

```python
import pytest
from tempo_app.core.chart_generator import ChartGenerator

def test_code_extraction_with_markdown():
    """Test extracting code from markdown-wrapped response."""
    response = """Here's the code:
    ```python
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3])
    ```
    """

    generator = ChartGenerator()
    code = generator._extract_code(response)

    assert "import matplotlib.pyplot as plt" in code
    assert "```" not in code  # Markdown removed

def test_safe_execution_blocks_file_io():
    """Test that file operations are blocked."""
    code = "open('/etc/passwd', 'r').read()"

    with pytest.raises(ChartGenerationError):
        generator.execute_code(code, pd.DataFrame(), Path("output.png"))

def test_schema_extraction():
    """Test DataFrame schema extraction."""
    df = pd.DataFrame({
        'TIME': pd.date_range('2024-01-01', periods=10, freq='H'),
        'NO2': [1.0, 2.0, 3.0] * 3 + [1.0],
        'site': ['BV'] * 10
    })

    schema = generator.get_dataframe_schema(df)

    assert 'TIME' in schema['columns']
    assert 'datetime64' in schema['dtypes']['TIME']
    assert 'BV' in schema['sample_values']['site']
```

### 2. Integration Tests

**Test file: `tests/test_ai_workflow.py`**

```python
def test_end_to_end_chart_generation(tmp_path):
    """Test full workflow from query to saved plot."""

    # Create test dataset
    df = create_test_dataframe()

    # Generate code
    generator = ChartGenerator()
    schema = generator.get_dataframe_schema(df)
    code = generator.generate_code("Plot NO2 by hour", schema)

    # Execute code
    output_path = tmp_path / "test_plot.png"
    plot_path = generator.execute_code(code, df, output_path)

    # Verify plot exists
    assert plot_path.exists()
    assert plot_path.stat().st_size > 1000  # Non-empty image
```

### 3. Manual Testing Checklist

- [ ] Configure Gemini API key in Settings
- [ ] Select a complete dataset
- [ ] Enter query: "Plot NO2 trends by hour"
- [ ] Verify code is generated and displayed
- [ ] Verify plot is created and visible
- [ ] Edit code (e.g., change title) and re-run
- [ ] Save analysis to database
- [ ] Load saved analysis from history
- [ ] Test error cases:
  - [ ] Invalid API key
  - [ ] Malformed query
  - [ ] Code with syntax error
  - [ ] Empty dataset

---

## Performance Optimization

### 1. DataFrame Downsampling

For large datasets (>100k rows), downsample before analysis:

```python
# In DataFrameConverter.dataset_to_dataframe()
if len(df) > 100_000:
    # Take every 10th point
    df = df.iloc[::10]
```

### 2. Lazy Loading

Don't load DataFrame until user clicks "Generate":

```python
async def _on_generate_clicked(self, e):
    if self.current_df is None:
        # Load on-demand
        await self._load_dataset()

    # Then generate code
    ...
```

### 3. Caching Gemini Responses

Cache identical queries to avoid redundant API calls:

```python
# In ChartGenerator
self._cache = {}  # {query_hash: code}

def generate_code(self, query, schema):
    cache_key = hash((query, frozenset(schema['columns'])))

    if cache_key in self._cache:
        return self._cache[cache_key]

    code = self._call_gemini(query, schema)
    self._cache[cache_key] = code
    return code
```

---

## Example Queries

Here are queries users can try:

### Time Series
- "Plot NO₂ levels over time at BV site"
- "Show HCHO trends for the past week"
- "Compare NO₂ at BV and LA sites on the same chart"

### Aggregations
- "Plot average NO₂ by hour of day"
- "Show median HCHO levels by day of week"
- "Create a bar chart of NO₂ averages for each site"

### Correlations
- "Scatter plot of NO₂ vs HCHO"
- "Plot FNR (HCHO/NO₂ ratio) over time"

### Statistical
- "Box plot of NO₂ distribution by site"
- "Histogram of HCHO values"
- "Show NO₂ quantiles (25th, 50th, 75th) by hour"

### Filtering
- "Plot NO₂ only on weekdays"
- "Show HCHO during rush hours (7-9 AM and 5-7 PM)"
- "Compare NO₂ on weekends vs weekdays"

---

## Deployment Checklist

### Phase 1: Core Implementation
- [ ] Add `google-generativeai` to requirements.txt
- [ ] Create `core/chart_generator.py`
- [ ] Create `core/df_converter.py`
- [ ] Add Analysis model to `storage/models.py`
- [ ] Update `storage/database.py` with analysis methods
- [ ] Update `core/config.py` with gemini_api_key

### Phase 2: UI Implementation
- [ ] Create `ui/pages/ai_analysis.py`
- [ ] Update `ui/pages/settings.py` with API key field
- [ ] Update `ui/shell.py` with navigation

### Phase 3: Testing
- [ ] Unit tests for ChartGenerator
- [ ] Integration tests for workflow
- [ ] Manual testing with real datasets
- [ ] Security audit of code execution

### Phase 4: Documentation
- [ ] Update README with AI Analysis feature
- [ ] Add example queries to user guide
- [ ] Document API key setup process

---

## Future Enhancements

### Phase 3: Advanced Features
- [ ] **Multi-chart layouts**: Generate subplots (2x2 grid)
- [ ] **Interactive plots**: Use Plotly instead of matplotlib
- [ ] **Data export**: Export filtered data to CSV
- [ ] **Code templates**: Pre-built chart templates for common queries
- [ ] **Code history**: Track all code versions for an analysis

### Phase 4: Collaboration
- [ ] **Share analyses**: Export analysis (code + plot + query) as JSON
- [ ] **Import analyses**: Load shared analyses from other users
- [ ] **Analysis templates**: Build library of common analyses

### Phase 5: Advanced AI
- [ ] **Multi-step queries**: "First filter weekdays, then plot NO₂ vs time"
- [ ] **Anomaly detection**: "Highlight days with unusually high NO₂"
- [ ] **Statistical summaries**: "Summarize NO₂ statistics in text"

---

## Cost Estimation

### Gemini API Pricing (as of 2024)

**gemini-1.5-flash** (recommended):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Typical query cost:**
- Prompt: ~1,000 tokens (DataFrame schema + instructions)
- Response: ~200 tokens (generated code)
- **Cost per query: ~$0.0001 (0.01 cents)**

**Free tier:**
- 15 requests per minute
- 1 million tokens per day
- **~10,000 free queries per day**

**Budget estimate for 1000 users:**
- Average 10 queries/user/day = 10,000 queries/day
- Cost: ~$1/day = **$30/month**

---

## Conclusion

This implementation provides a **lightweight, transparent, and reproducible** AI chart generation system:

✅ **Lightweight**: No heavy dependencies, just direct Gemini API
✅ **Transparent**: Users see and can edit all generated code
✅ **Reproducible**: Every plot is saved with its source code
✅ **Safe**: Sandboxed code execution prevents malicious code
✅ **Cost-effective**: Free tier covers most use cases

**Next Steps:**
1. Get Gemini API key: https://makersuite.google.com/app/apikey
2. Implement core components (chart_generator.py, database model)
3. Build UI page (ai_analysis.py)
4. Test with real TEMPO datasets
5. Deploy and gather user feedback

**Estimated implementation time**: 2-3 days for a working prototype
