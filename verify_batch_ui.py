
import asyncio
import flet as ft
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from tempo_app.storage.models import BatchJob, BatchSite, BatchJobStatus, BatchSiteStatus, Dataset, BoundingBox
from tempo_app.core.batch_scheduler import StatusEvent, StatusLevel
from tempo_app.ui.pages.batch_import import BatchImportPage
from tempo_app.ui.pages.library import LibraryPage

async def main(page: ft.Page):
    page.title = "Batch & Library Verification"
    page.theme_mode = ft.ThemeMode.DARK
    
    # --- MOCK DATA ---
    mock_db = MagicMock()
    mock_job_id = "job-123"
    mock_job = BatchJob(
        id=mock_job_id,
        name="Test Batch Job",
        created_at=datetime.now(),
        status=BatchJobStatus.RUNNING,
        total_sites=10,
        completed_sites=0,
        failed_sites=0,
        batch_size=5
    )
    
    sites = [
        BatchSite(id=i, batch_job_id=mock_job_id, site_name=f"Site_{i}", sequence_number=i, latitude=0, longitude=0)
        for i in range(1, 11)
    ]
    
    # Mock datasets for Library
    datasets = []
    # 3 in batch
    for i in range(1, 4):
        datasets.append(Dataset(
            id=f"ds_{i}", 
            name=f"Site_{i}", 
            created_at=datetime.now(),
            batch_job_id=mock_job_id,
            bbox=BoundingBox(0,0,0,0),
            date_start=datetime.now().date(),
            date_end=datetime.now().date(),
            day_filter=[],
            hour_filter=[],
            max_cloud=0.0,
            max_sza=0.0
        ))
    # 2 independent
    for i in range(4, 6):
        datasets.append(Dataset(
            id=f"ds_{i}", 
            name=f"Independent_{i}", 
            created_at=datetime.now(),
            bbox=BoundingBox(0,0,0,0),
            date_start=datetime.now().date(),
            date_end=datetime.now().date(),
            day_filter=[],
            hour_filter=[],
            max_cloud=0.0,
            max_sza=0.0
        ))
        
    mock_db.get_all_datasets.return_value = datasets

    # --- TABS ---
    # --- LAYOUT ---
    # Simplified to avoid Tab API issues
    page.add(
        ft.Column([
            ft.Text("Batch Import UI Inspection", size=20, weight=ft.FontWeight.BOLD),
            ft.Container(extract_batch_page(page, mock_db, mock_job, sites), height=500, border=ft.border.all(1, ft.Colors.GREY)),
            ft.Divider(),
            ft.Text("Library UI Inspection", size=20, weight=ft.FontWeight.BOLD),
            ft.Container(extract_library_page(page, mock_db), height=500, border=ft.border.all(1, ft.Colors.GREY)),
        ], expand=True, scroll=ft.ScrollMode.AUTO)
    )

def extract_batch_page(page, db, job, sites):
    # We need to instantiate BatchImportPage but inject our mock scheduler interaction
    # Since BatchImportPage creates its own scheduler, we might need to monkeypatch or just use the UI components directly?
    # Let's just create the page and trigger _on_log manually.
    
    batch_page = BatchImportPage(db)
    batch_page._current_job = job
    
    # Simulate processing
    def start_sim(e):
        page.run_task(simulate_batch, batch_page, job, sites)
        
    btn = ft.ElevatedButton("Start Mock Import", on_click=start_sim)
    
    return ft.Column([btn, batch_page])

def extract_library_page(page, db):
    lib_page = LibraryPage(db)
    return lib_page

async def simulate_batch(batch_page, job, sites):
    # Simulate 5 concurrent sites
    active_indices = [0, 1, 2, 3, 4]
    progress = {i: 0.0 for i in active_indices}
    
    # Loop until all sites done (simplified)
    for _ in range(50):
        await asyncio.sleep(0.1)
        
        for i in active_indices:
            site = sites[i]
            progress[i] += 0.05
            
            # Emit Progress Event
            event = StatusEvent(level=StatusLevel.PROGRESS, message=f"Downloading... {int(progress[i]*100)}%", progress=progress[i])
            batch_page._on_log(job, site, event)
            
            if progress[i] >= 1.0:
                # Complete
                batch_page._on_site_complete(site)
                # Pick next available site? For test just reset or stop
                # cycle
                progress[i] = 0.0 # Restart for visual effect or pick next
                
                # If we had a real queue we'd pick next. 
                # Let's just finish site 0 and start site 5
                if i == 0 and 5 not in active_indices:
                    active_indices.remove(0)
                    active_indices.append(5)
                    progress[5] = 0.0

ft.app(target=main)
