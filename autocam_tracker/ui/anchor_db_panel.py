from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class AnchorDBPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, on_track_gid: Callable[[int], None] = None):
        super().__init__(parent)
        self.on_track_gid = on_track_gid

        # UI Layout
        lbl_title = ttk.Label(self, text="Hierarchical Anchor DB", font=("Segoe UI", 12, "bold"))
        lbl_title.pack(side="top", anchor="w", padx=5, pady=5)

        # Treeview to display GIDs and their anchored shots
        self.tree = ttk.Treeview(self, columns=("status",), show="tree headings", height=8)
        self.tree.heading("#0", text="Global ID / Shots")
        self.tree.heading("status", text="Status")
        self.tree.column("#0", width=150)
        self.tree.column("status", width=80)
        self.tree.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        
        self.active_gids = set()
        self._is_updating = False

    def update_db_view(self, db_state: dict):
        """
        Updates the treeview with current DB state.
        db_state format: { gid: {"label": str, "shots": [shot_id, ...]} }
        """
        self._is_updating = True
        try:
            # Clear old items not in db
            for item in self.tree.get_children():
                if item.startswith("GID_"):
                    gid = int(item.split("_")[1])
                    if gid not in db_state:
                        self.tree.delete(item)

            # Update existing or add new
            for gid, info in db_state.items():
                item_id = f"GID_{gid}"
                label = info.get("label", "car")
                total = info.get("total_features", 0)
                display_text = f"{label} G{gid} ({total} Feat)"
                shots = info.get("shots", {})
                
                if not self.tree.exists(item_id):
                    self.tree.insert("", "end", iid=item_id, text=display_text, open=True)
                else:
                    self.tree.item(item_id, text=display_text)
                    
                # Update shots children
                existing_shots = set()
                for child in self.tree.get_children(item_id):
                    existing_shots.add(int(child.split("_")[-1]))
                    
                for shot_id, count in shots.items():
                    child_id = f"{item_id}_shot_{shot_id}"
                    shot_text = f"Shot {shot_id} [{count}]"
                    if shot_id not in existing_shots:
                        self.tree.insert(item_id, "end", iid=child_id, text=shot_text, values=("Anchored",))
                    else:
                        self.tree.item(child_id, text=shot_text)
        finally:
            self._is_updating = False

    def get_selected_gid(self) -> int:
        selection = self.tree.selection()
        if not selection:
            return -1
        item_id = selection[0]
        if item_id.startswith("GID_"):
            return int(item_id.split("_")[1])
        elif "_shot_" in item_id:
            return int(item_id.split("_")[1])
        return -1

    def _on_tree_select(self, event):
        if not self.on_track_gid or self._is_updating:
            return
        # Use after idle to prevent UI block if multiple selections happen fast
        gid = self.get_selected_gid()
        if gid != -1:
            self.on_track_gid(gid)
