import os, shutil, glob, re, torch, random, sys, tifffile
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
from model import serialization
from utilities import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QStatusBar, QScrollArea, QSlider, QGroupBox, QFormLayout,
                             QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QCursor, QPen, QColor, QBrush, QPolygonF
from PyQt5.QtCore import Qt, QRect, QEvent, QPointF
from functionalities.inference import GlobalConsensusInference

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
torch.serialization.add_safe_globals([defaultdict, list])


def get_global_coords(y, x, H_pad, W_pad, flip_code, crop_size):
    """Calculates global bounding coordinates after considering spatial flip augmentations."""
    if flip_code is None: return y, x
    if flip_code == 1:    return y, W_pad - x - crop_size
    if flip_code == 0:    return H_pad - y - crop_size, x
    if flip_code == -1:   return H_pad - y - crop_size, W_pad - x - crop_size
    return y, x


def collect_files(root):
    """Finds and indexes source image paths and prior instance masks across versioned folders."""
    abs_root = os.path.abspath(root)
    parent_dir = os.path.dirname(abs_root)
    base_name = os.path.basename(abs_root)
    existing_versions = glob.glob(os.path.join(parent_dir, f"{base_name}_v*"))
    highest_version = 0
    latest_version_path = None
    if existing_versions:
        for directory in existing_versions:
            match = re.search(r'_v(\d+)$', directory)
            if match:
                ver = int(match.group(1))
                if ver > highest_version:
                    highest_version = ver
                    latest_version_path = directory
    if latest_version_path:
        img_dir = os.path.join(latest_version_path, "source_images")
        mask_dir = os.path.join(latest_version_path, "source_masks")
        logging.prog_state(f"[*] Resuming session. Loading data from prior version: v{highest_version}")
    else:
        img_dir = os.path.join(abs_root, "source_images")
        mask_dir = os.path.join(abs_root, "source_masks")
        logging.prog_state("[*] No prior versions found. Starting fresh from base root.")
    file_dict = {}
    valid_exts = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF')
    if not os.path.exists(img_dir):
        logging.prog_state(f"[!] Error: Directory '{img_dir}' does not exist.")
        return file_dict
    os.makedirs(mask_dir, exist_ok=True)
    for filename in sorted(os.listdir(img_dir)):
        if filename.lower().endswith(valid_exts):
            name_no_ext, ext = os.path.splitext(filename)
            mask_filename = f"{name_no_ext}_mask.png"
            img_path = os.path.abspath(os.path.join(img_dir, filename))
            mask_path = os.path.abspath(os.path.join(mask_dir, mask_filename))
            file_dict[filename] = {
                "img": img_path,
                "mask": mask_path if os.path.exists(mask_path) else None
            }
    logging.prog_state(f"Collection complete. Total images found: {len(file_dict)}")
    return file_dict


def create_versioned_root(root):
    """Creates a new versioned folder hierarchy for source images and source masks."""
    abs_root = os.path.abspath(root)
    parent_dir = os.path.dirname(abs_root)
    base_name = os.path.basename(abs_root)
    existing_versions = glob.glob(os.path.join(parent_dir, f"{base_name}_v*"))
    version = 1
    if existing_versions:
        version_nums = []
        for directory in existing_versions:
            match = re.search(r'_v(\d+)$', directory)
            if match:
                version_nums.append(int(match.group(1)))
        if version_nums:
            version = max(version_nums) + 1
    new_root_path = os.path.join(parent_dir, f"{base_name}_v{version}")
    os.makedirs(new_root_path, exist_ok=True)
    os.makedirs(os.path.join(new_root_path, "source_images"), exist_ok=True)
    os.makedirs(os.path.join(new_root_path, "source_masks"), exist_ok=True)
    logging.prog_state("Successfully generated new dataset version directory:")
    logging.prog_state(f" └─ {new_root_path}")
    return new_root_path, os.path.join(new_root_path, "source_images"), os.path.join(new_root_path, "source_masks")


class LISNetCanvas(QWidget):
    def __init__(self, parent=None):
        """Initializes canvas tracking variables, cursor styles, and mouse event states."""
        super().__init__(parent)
        self.parent = parent
        self.pixmap = None
        self.hover_pos = None
        self.zoom = 1.0
        self.is_lmb_pressed = False
        self.dragged = False
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.NoContextMenu)

    def set_data(self, pixmap, zoom=1.0):
        """Updates base pixmap reference and scales the canvas widget size."""
        self.pixmap = pixmap
        self.zoom = float(zoom)
        if self.parent and self.parent.mask is not None:
            h, w = self.parent.mask.shape[:2]
            self.setFixedSize(int(w * self.zoom), int(h * self.zoom))
        elif self.pixmap and not self.pixmap.isNull():
            self.setFixedSize(int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom))
        else:
            self.setFixedSize(int(512 * self.zoom), int(512 * self.zoom))
        self.update()

    def paintEvent(self, event):
        """Renders rasterized base pixmap, clean boundary lines, and live cursor guidance without dots."""
        if not self.pixmap: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawPixmap(self.rect(), self.pixmap)

        if self.parent.current_poly:
            is_void = self.parent.is_drawing_void
            line_color = QColor(200, 200, 200) if is_void else QColor(0, 220, 255)
            
            painter.setPen(QPen(line_color, max(1, int(self.zoom * 0.75)), Qt.SolidLine))
            screen_pts = [QPointF((pt[0] + 0.5) * self.zoom, (pt[1] + 0.5) * self.zoom) for pt in self.parent.current_poly]
            
            for i in range(len(screen_pts) - 1):
                painter.drawLine(screen_pts[i], screen_pts[i+1])
                
            if not self.is_lmb_pressed and self.hover_pos is not None:
                painter.setPen(QPen(line_color, max(1, int(self.zoom * 0.5)), Qt.DashLine))
                painter.drawLine(screen_pts[-1], self.hover_pos)

    def mousePressEvent(self, event):
        """Routes mouse clicks to continuous drag drawing, selection, deletion, or inference."""
        if self.parent is None or self.parent.mask is None:
            return
        
        pos = event.pos()
        x = int(pos.x() // self.zoom)
        y = int(pos.y() // self.zoom)
        h, w = self.parent.mask.shape[:2]
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)

        mods = event.modifiers()
        is_void_mod = bool(mods & Qt.ShiftModifier)
        is_select = bool(mods & Qt.ControlModifier)
        is_delete = bool(mods & Qt.AltModifier)

        if event.button() == Qt.RightButton:
            if self.parent.current_poly:
                self.parent.close_and_rasterize_polygon(is_erase=True)
            return

        if event.button() == Qt.LeftButton:
            if self.parent.is_prompting_mode and not (is_void_mod or is_select or is_delete):
                self.parent.run_point_inference(x, y)
                return

            if is_delete:
                self.parent.delete_instance_at(x, y)
                return

            if is_select:
                self.parent.pick_instance_at(x, y)
                return

            self.is_lmb_pressed = True
            self.dragged = False
            self.parent.add_polygon_anchor(x, y, is_void=is_void_mod)

    def mouseMoveEvent(self, event):
        """Appends pixel anchors during LMB drag and repaints canvas preview markers."""
        self.hover_pos = event.pos()
        if self.is_lmb_pressed and self.parent is not None and self.parent.mask is not None:
            self.dragged = True
            x = int(event.pos().x() // self.zoom)
            y = int(event.pos().y() // self.zoom)
            h, w = self.parent.mask.shape[:2]
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
            mods = event.modifiers()
            is_void_mod = bool(mods & Qt.ShiftModifier)
            self.parent.add_polygon_anchor(x, y, is_void=is_void_mod)
        elif self.parent.current_poly:
            self.update()

    def mouseReleaseEvent(self, event):
        """Closes and rasterizes polygon if drag stroke finished with valid anchors."""
        if event.button() == Qt.LeftButton:
            self.is_lmb_pressed = False
            if self.dragged and len(self.parent.current_poly) >= 3:
                self.parent.close_and_rasterize_polygon(is_erase=False)
            self.dragged = False
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Finalizes active polygon vertex sequence on double click."""
        if event.button() == Qt.LeftButton and self.parent.current_poly:
            self.parent.close_and_rasterize_polygon(is_erase=False)

    def leaveEvent(self, event):
        """Clears local cursor cache and resets mouse press flags when cursor exits."""
        self.hover_pos = None
        self.is_lmb_pressed = False
        self.dragged = False
        if self.parent.current_poly:
            self.update()


class LISNetLabeler(QMainWindow):
    def __init__(self, file_dict, source_img_dir, source_mask_dir, model, crop_size, device):
        """Sets up main window configuration, data structures, and UI element hierarchy."""
        super().__init__()
        self.model = model.to(device).eval()
        self.crop_size = crop_size
        self.device = device
        self.file_dict = file_dict
        self.filenames = [f for f in list(file_dict.keys()) if not f.startswith("done_")]
        self.source_img_dir = source_img_dir
        self.source_mask_dir = source_mask_dir
        self.current_idx = 0
        self.active_id = 1
        self.zoom_factor = 1.0
        self.hide_mask = False
        self.hide_void = False
        self.is_prompting_mode = False
        self.img_rgb = None
        self.mask = None
        self.void_mask = None
        self.current_poly = []
        self.is_drawing_void = False
        self._init_colormaps()
        self.history = []
        self.ai_added_ids = set()
        self.init_ui()
        self.apply_styles()
        if self.filenames:
            self.load_sample(0)

    def apply_styles(self):
        """Applies application-wide CSS styling across widgets."""
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QGroupBox {
                border: 2px solid #333333;
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
                text-transform: uppercase;
                color: #888888;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #333;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #444; }
            QStatusBar { background-color: #000; color: #00ff00; font-family: 'Consolas'; }
            QLabel b { color: #00d4ff; } 
        """)

    def init_ui(self):
        """Instantiates layout groups, interactive controls, and canvas wrappers."""
        self.setWindowTitle("🍇 LISNet Labeler (Polygon Tool)")
        self.setGeometry(100, 100, 1500, 1000)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_sidebar = QVBoxLayout()
        left_sidebar.setContentsMargins(10, 10, 10, 10)
        left_sidebar.setSpacing(15)
        
        data_group = QGroupBox("Dataset Navigation")
        data_layout = QVBoxLayout()
        self.file_list = QComboBox()
        self.file_list.addItems(self.filenames)
        self.file_list.currentIndexChanged.connect(self.load_sample)
        data_layout.addWidget(QLabel("Current Image:"))
        data_layout.addWidget(self.file_list)
        data_group.setLayout(data_layout)
        left_sidebar.addWidget(data_group)

        inst_group = QGroupBox("Instance Status")
        inst_layout = QVBoxLayout()
        self.lbl_active_id = QLabel("Active Instance ID: 1")
        self.lbl_active_id.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff88; padding: 4px;")
        inst_layout.addWidget(self.lbl_active_id)

        self.btn_commit_poly = QPushButton("Complete Polygon (Enter)")
        self.btn_commit_poly.clicked.connect(lambda: self.close_and_rasterize_polygon(is_erase=False))
        inst_layout.addWidget(self.btn_commit_poly)

        self.btn_discard_poly = QPushButton("Cancel Polygon (Esc)")
        self.btn_discard_poly.clicked.connect(self.cancel_active_polygon)
        inst_layout.addWidget(self.btn_discard_poly)

        inst_group.setLayout(inst_layout)
        left_sidebar.addWidget(inst_group)

        legend_group = QGroupBox("Control Scheme")
        legend_layout = QFormLayout()
        legend_layout.addRow("LMB Drag:", QLabel("<b>Draw Contour (Snaps to Px)</b>"))
        legend_layout.addRow("LMB Click:", QLabel("<b>Add Anchor</b>"))
        legend_layout.addRow("Enter / Release:", QLabel("<b>Fill Polygon</b>"))
        legend_layout.addRow("R-Click (in poly):", QLabel("<b>Erase Polygon</b>"))
        legend_layout.addRow("Esc:", QLabel("<b>Cancel Anchor Path</b>"))
        legend_layout.addRow("Shift + Drag:", QLabel("<b>Void (255) Layer</b>"))
        legend_layout.addRow("Ctrl + L-Click:", QLabel("<b>Pick ID</b>"))
        legend_layout.addRow("Alt + L-Click:", QLabel("<b>Delete Instance</b>"))
        legend_layout.addRow("Ctrl + Wheel:", QLabel("<b>Zoom Viewport</b>"))
        legend_group.setLayout(legend_layout)
        left_sidebar.addWidget(legend_group)
        left_sidebar.addStretch()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: #121212; }
            QScrollBar:horizontal { background: #1e1e1e; height: 12px; }
            QScrollBar:handle:horizontal { background: #444; border-radius: 6px; }
            QScrollBar:vertical { background: #1e1e1e; width: 12px; }
            QScrollBar:handle:vertical { background: #444; border-radius: 6px; }
        """)
        self.scroll_area.setFocusPolicy(Qt.NoFocus)
        self.scroll_area.viewport().setFocusPolicy(Qt.NoFocus)
        self.scroll_area.installEventFilter(self)
        self.scroll_area.viewport().installEventFilter(self)
        self.canvas = LISNetCanvas(self)
        self.canvas.setFocusPolicy(Qt.NoFocus)
        self.scroll_area.setWidget(self.canvas)

        right_sidebar = QVBoxLayout()
        right_sidebar.setContentsMargins(10, 10, 10, 10)
        right_sidebar.setSpacing(15)

        view_group = QGroupBox("Layer Controls")
        view_layout = QVBoxLayout()
        self.btn_toggle_mask = QPushButton("Toggle Instances")
        self.btn_toggle_mask.clicked.connect(self.toggle_mask_visibility)
        view_layout.addWidget(self.btn_toggle_mask)
        self.btn_toggle_void = QPushButton("Toggle Void Layer")
        self.btn_toggle_void.clicked.connect(self.toggle_void_visibility)
        view_layout.addWidget(self.btn_toggle_void)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 3000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        view_layout.addWidget(QLabel("Zoom:"))
        view_layout.addWidget(self.zoom_slider)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_display)
        view_layout.addWidget(QLabel("Brightness:"))
        view_layout.addWidget(self.brightness_slider)
        view_group.setLayout(view_layout)
        right_sidebar.addWidget(view_group)

        ai_group = QGroupBox("Intelligence")
        ai_layout = QVBoxLayout()
        self.btn_prompt_mode = QPushButton("Toggle Prompting: OFF")
        self.btn_prompt_mode.clicked.connect(self.toggle_prompt_mode)
        ai_layout.addWidget(self.btn_prompt_mode)
        self.btn_auto = QPushButton("Full Image Inference")
        self.btn_auto.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold;")
        self.btn_auto.clicked.connect(self.run_full_inference)
        ai_layout.addWidget(self.btn_auto)
        ai_group.setLayout(ai_layout)
        right_sidebar.addWidget(ai_group)
        right_sidebar.addStretch()

        self.cb_test_split = QCheckBox("Include Test Set (70/15/15 vs 80/20)")
        self.cb_test_split.setChecked(True)
        self.cb_test_split.setStyleSheet("color: #e0e0e0; font-weight: bold; padding: 5px;")
        right_sidebar.addWidget(self.cb_test_split)
        self.btn_undo = QPushButton("UNDO ACTION")
        self.btn_undo.setObjectName("UndoButton")
        self.btn_undo.clicked.connect(self.undo)
        right_sidebar.addWidget(self.btn_undo)
        self.btn_save = QPushButton("SAVE AND NEXT")
        self.btn_save.setObjectName("SaveButton")
        self.btn_save.clicked.connect(self.save_and_next)
        right_sidebar.addWidget(self.btn_save)
        self.btn_done = QPushButton("MARK AS DONE")
        self.btn_done.setObjectName("DoneButton")
        self.btn_done.clicked.connect(self.mark_as_done)
        right_sidebar.addWidget(self.btn_done)
        self.btn_finish = QPushButton("CONCLUDE SESSION")
        self.btn_finish.setObjectName("FinishButton")
        self.btn_finish.clicked.connect(self.close)
        right_sidebar.addWidget(self.btn_finish)

        main_layout.addLayout(left_sidebar, 1)
        main_layout.addWidget(self.scroll_area, 6)
        main_layout.addLayout(right_sidebar, 1)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def eventFilter(self, watched, event):
        """Intercepts viewport events to enforce custom keyboard shortcuts and smooth zooming."""
        if event.type() == QEvent.Wheel:
            if QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.wheelEvent(event)
                return True 
        elif event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(watched, event)

    def set_active_id(self, new_val):
        """Updates internal active instance ID and displays the synchronized value on the UI."""
        self.active_id = int(new_val)
        self.lbl_active_id.setText(f"Active Instance ID: {self.active_id}")

    def toggle_prompt_mode(self):
        """Switches interactive mode between manual polygon vertex anchoring and model point queries."""
        self.is_prompting_mode = not self.is_prompting_mode
        if self.is_prompting_mode:
            self.cancel_active_polygon()
            self.btn_prompt_mode.setText("Toggle Prompting: ON")
            self.btn_prompt_mode.setStyleSheet("background-color: #e59d04; color: black; font-weight: bold;")
            self.status.showMessage("Prompting Mode ON: Left click on a leaf to query the AI.")
        else:
            self.btn_prompt_mode.setText("Toggle Prompting: OFF")
            self.btn_prompt_mode.setStyleSheet("background-color: #333; color: white;")
            self.status.showMessage("Prompting Mode OFF: Polygon tool active.")

    def toggle_mask_visibility(self):
        """Toggles visual rendering of foreground instance segments."""
        self.hide_mask = not self.hide_mask
        self.btn_toggle_mask.setText("Instances Hidden" if self.hide_mask else "Toggle Instances")
        self.update_display()

    def toggle_void_visibility(self):
        """Toggles visual rendering of the void background boundary mask."""
        self.hide_void = not self.hide_void
        self.btn_toggle_void.setText("Void Layer Hidden" if self.hide_void else "Toggle Void Layer")
        self.update_display()

    def add_polygon_anchor(self, x, y, is_void=False):
        """Appends distinct pixel coordinates sequentially to build the boundary trace."""
        if not self.current_poly:
            self.is_drawing_void = is_void

        if self.current_poly and self.current_poly[-1] == (x, y):
            return

        self.current_poly.append((x, y))
        mode_text = "Void" if self.is_drawing_void else f"Instance {self.active_id}"
        self.status.showMessage(f"Trace node #{len(self.current_poly)} at ({x}, {y}) [{mode_text}].")
        self.canvas.update()

    def cancel_active_polygon(self):
        """Clears all uncommitted anchor vertices currently stored on the canvas."""
        if self.current_poly:
            self.current_poly.clear()
            self.canvas.update()
            self.status.showMessage("Polygon drawing canceled.")

    def close_and_rasterize_polygon(self, is_erase=False):
        """Renders polygon vertex sequence onto the 2D pixel mask and advances active instance ID."""
        if len(self.current_poly) < 3:
            self.status.showMessage("Need at least 3 points to rasterize a polygon.")
            return

        self.save_state_for_undo()
        pts = np.array(self.current_poly, dtype=np.int32).reshape((-1, 1, 2))
        
        if self.is_drawing_void:
            fill_val = 0 if is_erase else 255
            cv2.fillPoly(self.void_mask, [pts], int(fill_val))
            action_desc = "Erased from" if is_erase else "Filled into"
            self.status.showMessage(f"{action_desc} Void layer with fine-resolution polygon.")
        else:
            fill_val = 0 if is_erase else self.active_id
            cv2.fillPoly(self.mask, [pts], int(fill_val))
            action_desc = "Erased from" if is_erase else "Created"
            self.status.showMessage(f"{action_desc} Instance ID {self.active_id} via fine-resolution polygon.")
            if not is_erase:
                self.set_active_id(self.get_next_free_id())

        self.current_poly.clear()
        self.update_display()

    def pick_instance_at(self, x, y):
        """Extracts existing instance ID at coordinates and sets it as the active drawing ID."""
        current_val = self.mask[y, x]
        if 0 < current_val < 255:
            self.set_active_id(current_val)
            self.status.showMessage(f"Selected existing ID {current_val} via Pick Tool.")
        else:
            self.status.showMessage("No valid instance to pick here.")

    def delete_instance_at(self, x, y):
        """Globally purges all pixels mapped to the target instance ID found under cursor."""
        current_val = self.mask[y, x]
        if 0 < current_val < 255:
            self.save_state_for_undo()
            self.mask[self.mask == current_val] = 0
            self.update_display()
            self.set_active_id(self.get_next_free_id())
            self.status.showMessage(f"Globally deleted instance ID {current_val}")

    def closeEvent(self, event):
        """Exports finalized dataset versions and partition splits prior to application exit."""
        self.status.showMessage("Finalizing dataset... Exporting files.")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if self.img_rgb is not None and self.mask is not None and self.filenames:
            try:
                fname = self.filenames[self.current_idx]
                out_img = os.path.join(self.source_img_dir, fname)
                out_mask = os.path.join(self.source_mask_dir, f"{os.path.splitext(fname)[0]}_mask.png")
                merged_mask = self.mask.copy()
                merged_mask[(self.mask == 0) & (self.void_mask == 255)] = 255
                cv2.imwrite(out_img, cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(out_mask, merged_mask.astype(np.uint8))
            except Exception as e:
                print(f"Warning: Failed to auto-save current image: {e}")
        try:
            for fname in self.file_dict.keys():
                save_path_img = os.path.join(self.source_img_dir, fname)
                data = self.file_dict[fname]
                if not os.path.exists(save_path_img):
                    shutil.copy2(data['img'], save_path_img)
                    mask_name = f"{os.path.splitext(fname)[0]}_mask.png"
                    save_path_mask = os.path.join(self.source_mask_dir, mask_name)
                    if data['mask'] and os.path.exists(data['mask']):
                        shutil.copy2(data['mask'], save_path_mask)
                    else:
                        img_shape = cv2.imread(data['img']).shape
                        blank_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint16)
                        cv2.imwrite(save_path_mask, blank_mask)
            self.create_final_split()
            split_msg = "(70/15/15)" if self.cb_test_split.isChecked() else "(80/20)"
            logging.prog_state(f"Dataset consolidated and split {split_msg}.")
        except Exception as e:
            print(f"Finalization Error: {e}")
            self.status.showMessage("Error during finalization.")
        finally:
            QApplication.restoreOverrideCursor()
            event.accept()

    def create_final_split(self):
        """Slices annotated full-resolution images into training patches and distributes splits."""
        random.seed(42)
        all_large_images = sorted(os.listdir(self.source_img_dir))
        content_crops = []
        background_crops = []
        win_size = self.crop_size
        stride = self.crop_size // 2
        source_folder = os.path.dirname(self.source_img_dir)  
        dataset_root = os.path.join(source_folder, "dataset") 
        temp_img_dir = os.path.join(source_folder, "temp_sliced_images")
        temp_mask_dir = os.path.join(source_folder, "temp_sliced_masks")
        os.makedirs(temp_img_dir, exist_ok=True)
        os.makedirs(temp_mask_dir, exist_ok=True)
        self.status.showMessage("Slicing large images into crops...")
        for f in all_large_images:
            name_no_ext, ext = os.path.splitext(f)
            m_name = f"{name_no_ext}_mask.png"
            img_path = os.path.join(self.source_img_dir, f)
            mask_path = os.path.join(self.source_mask_dir, m_name)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue
            img_h, img_w = img.shape[:2]
            y_coords = list(range(0, max(1, img_h - win_size + 1), stride))
            if img_h > win_size and y_coords[-1] != img_h - win_size:
                y_coords.append(img_h - win_size)
            x_coords = list(range(0, max(1, img_w - win_size + 1), stride))
            if img_w > win_size and x_coords[-1] != img_w - win_size:
                x_coords.append(img_w - win_size)
            for y in y_coords:
                for x in x_coords:
                    crop_img = img[y:y+win_size, x:x+win_size]
                    crop_mask = mask[y:y+win_size, x:x+win_size]
                    crop_name = f"{name_no_ext}_y{y}_x{x}{ext}"
                    if not np.any((crop_mask > 0) & (crop_mask < 255)):
                        background_crops.append(crop_name)
                        continue
                    resized_img = crop_img
                    resized_mask = crop_mask
                    crop_mask_name = f"{name_no_ext}_y{y}_x{x}_mask.png"
                    crop_img_path = os.path.join(temp_img_dir, crop_name)
                    crop_mask_path = os.path.join(temp_mask_dir, crop_mask_name)
                    cv2.imwrite(crop_img_path, resized_img)
                    cv2.imwrite(crop_mask_path, resized_mask)
                    if np.any((resized_mask > 0) & (resized_mask < 255)):
                        content_crops.append(crop_name)
                    else:
                        background_crops.append(crop_name)
        logging.prog_state(f"Generated {len(content_crops)} crops with valid instances.")
        logging.prog_state(f"Generated {len(background_crops)} background-only crops. Discarding all of them.")
        final_dataset = content_crops
        random.shuffle(final_dataset)
        n = len(final_dataset)
        include_test_set = self.cb_test_split.isChecked()
        if include_test_set:
            train_end = int(n * 0.70)
            val_end = train_end + int(n * 0.15)
            split_map = {
                "train": final_dataset[:train_end],
                "val": final_dataset[train_end:val_end],
                "test": final_dataset[val_end:]
            }
            self.status.showMessage("Distributing crops into train/val/test splits (70/15/15)...")
        else:
            train_end = int(n * 0.80)
            split_map = {
                "train": final_dataset[:train_end],
                "val": final_dataset[train_end:]
            }
            self.status.showMessage("Distributing crops into train/val splits (80/20)...")
        for split_name, files in split_map.items():
            img_dest = os.path.join(dataset_root, split_name, "images")
            mask_dest = os.path.join(dataset_root, split_name, "masks")
            os.makedirs(img_dest, exist_ok=True)
            os.makedirs(mask_dest, exist_ok=True)
            for f in files:
                m_name = f"{os.path.splitext(f)[0]}_mask.png"
                shutil.move(os.path.join(temp_img_dir, f), os.path.join(img_dest, f))
                shutil.move(os.path.join(temp_mask_dir, m_name), os.path.join(mask_dest, m_name))
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_mask_dir)
        logging.prog_state(f"Slicing complete. Dataset exported to: {dataset_root}")

    def _preprocess(self, img_rgb):
        """Standardizes color channels and normalizes image tensors for inference pipelines."""
        img = (img_rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    def keyPressEvent(self, event):
        """Handles keyboard shortcut events for navigation, undo, and polygon submission."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.current_poly:
                self.close_and_rasterize_polygon(is_erase=False)
            else:
                self.save_and_next()
        elif event.key() == Qt.Key_Left:
            new_idx = max(0, self.current_idx - 1)
            self.file_list.setCurrentIndex(new_idx)
        elif event.key() == Qt.Key_Right:
            new_idx = min(len(self.filenames) - 1, self.current_idx + 1)
            self.file_list.setCurrentIndex(new_idx)
        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undo()
        elif event.key() == Qt.Key_Escape:
            if self.current_poly:
                self.cancel_active_polygon()
            else:
                self.close()

    def set_zoom(self, value):
        """Maps zoom slider values to viewport centers."""
        viewport = self.scroll_area.viewport()
        center_pos = viewport.rect().center()
        self.apply_zoom(value / 100.0, center_pos)

    def apply_zoom(self, new_zoom_val, focus_pos):
        """Calculates scroll offsets to anchor zooming around cursor positions."""
        old_zoom = self.zoom_factor
        new_zoom = float(new_zoom_val)
        if old_zoom == new_zoom:
            return
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        img_x = (h_bar.value() + focus_pos.x()) / old_zoom
        img_y = (v_bar.value() + focus_pos.y()) / old_zoom
        self.zoom_factor = new_zoom
        self.update_display()
        new_h = int((img_x * new_zoom) - focus_pos.x())
        new_v = int((img_y * new_zoom) - focus_pos.y())
        h_bar.setValue(new_h)
        v_bar.setValue(new_v)

    def wheelEvent(self, event):
        """Captures mouse wheel movement during Ctrl down events to dynamically rescale the image."""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 15 if delta > 0 else -15
            new_val = self.zoom_slider.value() + step
            new_val = max(10, min(3000, new_val))
            viewport = self.scroll_area.viewport()
            mouse_pos = viewport.mapFromGlobal(QCursor.pos())
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(new_val)
            self.zoom_slider.blockSignals(False)
            self.apply_zoom(new_val / 100.0, mouse_pos)
            event.accept()
        else:
            super().wheelEvent(event)

    def load_sample(self, index):
        """Loads and formats chosen raw imagery, parses existing masks, and sets the active canvas."""
        self.cancel_active_polygon()
        self.history.clear()
        self.ai_added_ids.clear()
        self.current_idx = index
        fname = self.filenames[index]
        data = self.file_dict[fname]
        ext = os.path.splitext(data['img'])[1].lower()
        if ext in ['.tif', '.tiff']:
            raw_img = tifffile.imread(data['img'])
            if raw_img.dtype == np.float32 or raw_img.dtype == np.float64:
                self.img_rgb = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)
            elif raw_img.dtype == np.uint16:
                self.img_rgb = np.clip((raw_img / 65535.0) * 255.0, 0, 255).astype(np.uint8)
            else:
                self.img_rgb = raw_img.astype(np.uint8)
        else:
            img_bgr = cv2.imread(data['img'])
            self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.img_rgb.shape[:2]
        if data['mask'] and os.path.exists(data['mask']):
            full_mask = cv2.imread(data['mask'], cv2.IMREAD_UNCHANGED)
            if len(full_mask.shape) == 3: full_mask = full_mask[:,:,0]
            self.mask = np.where(full_mask == 255, 0, full_mask).astype(np.uint16)
            self.void_mask = np.where(full_mask == 255, 255, 0).astype(np.uint16)
        else:
            self.mask = np.zeros((h, w), dtype=np.uint16)
            self.void_mask = np.zeros((h, w), dtype=np.uint16)
        next_free_id = self.get_next_free_id()
        self.set_active_id(next_free_id)
        self.brightness_slider.blockSignals(True)
        self.brightness_slider.setValue(0)
        self.brightness_slider.blockSignals(False)
        self.update_display()
        self.status.showMessage(f"Loaded {fname} [{w}x{h}]. Next active ID: {next_free_id}")

    def save_state_for_undo(self):
        """Pushes full copies of active masks onto undo history buffers."""
        if self.mask is not None and self.void_mask is not None:
            self.history.append((self.mask.copy(), self.void_mask.copy(), self.active_id))
            if len(self.history) > 20:
                self.history.pop(0)

    def undo(self):
        """Restores mask matrices to their prior snapshot state."""
        if self.history:
            self.mask, self.void_mask, saved_id = self.history.pop()
            self.set_active_id(saved_id)
            self.cancel_active_polygon()
            self.update_display()
            self.status.showMessage("Undo successful.")
        else:
            self.status.showMessage("Nothing to undo.")

    def run_point_inference(self, px, py):
        """Executes targeted point prompt inference matching LISNet's coordinate conventions."""
        if self.img_rgb is None:
            return
        self.status.showMessage(f"AI Prompting: Running inference at ({px}, {py})...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            H, W = self.img_rgb.shape[:2]
            c_size = self.crop_size
            x1 = max(0, min(px - c_size // 2, W - c_size))
            y1 = max(0, min(py - c_size // 2, H - c_size))
            x2 = x1 + c_size
            y2 = y1 + c_size
            rel_x = px - x1
            rel_y = py - y1
            crop = self.img_rgb[y1:y2, x1:x2]
            if getattr(self, "zoom", False):
                crop_input = cv2.resize(crop, (c_size * 2, c_size * 2), interpolation=cv2.INTER_LINEAR)
                in_h, in_w = c_size * 2, c_size * 2
                scale = 2.0
            else:
                crop_input = crop
                in_h, in_w = c_size, c_size
                scale = 1.0
            y_norm = float(rel_y * scale) / max(in_h - 1, 1)
            x_norm = float(rel_x * scale) / max(in_w - 1, 1)
            y_norm = max(0.0, min(1.0, y_norm))
            x_norm = max(0.0, min(1.0, x_norm))
            pt_tensor = torch.tensor([[y_norm, x_norm]], dtype=torch.float32, device=self.device)
            prompt_coords = [pt_tensor]
            img_t = self._preprocess(crop_input)
            with torch.no_grad():
                out_fields, pred_masks, score_logits, _ = self.model(img_t, prompt_coords=prompt_coords)
            mask_logits = pred_masks[0, 0:1].unsqueeze(0)
            mask_upscaled = F.interpolate(
                mask_logits, 
                size=(c_size, c_size), 
                mode="bilinear", 
                align_corners=False
            ).squeeze()
            prob_map = torch.sigmoid(mask_upscaled).detach().cpu().numpy()
            binary_mask = prob_map > 0.5
            clean_area = int(np.sum(binary_mask))
            if clean_area > 30:
                valid_space = (self.mask[y1:y2, x1:x2] == 0)
                final_insert_mask = binary_mask & valid_space
                if np.any(final_insert_mask):
                    self.save_state_for_undo()
                    target_id = self.active_id
                    self.mask[y1:y2, x1:x2][final_insert_mask] = target_id
                    self.ai_added_ids.add(target_id)
                    self.set_active_id(self.get_next_free_id())
                    self.status.showMessage(f"AI Prompting: Successfully created instance {target_id} (area: {clean_area} px).")
                else:
                    self.status.showMessage("AI Prompting: Generated mask overlaps existing instances completely.")
            else:
                max_prob = float(prob_map.max())
                self.status.showMessage(f"AI Prompting: No instance detected (area: {clean_area} px, max prob: {max_prob:.2f}).")
        except Exception as e:
            self.status.showMessage(f"Prompting Error: {str(e)}")
            print(f"Point Inference Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.update_display()

    def run_full_inference(self):
        """Runs sliding window neural network inference with test-time augmentation across the canvas."""
        if self.img_rgb is None: 
            return
        self.save_state_for_undo()
        self.status.showMessage("AI Inference: Running Sliding Window + TTA Consensus... Please wait.")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            H_orig, W_orig = self.img_rgb.shape[:2]
            c_h = self.crop_size
            c_w = self.crop_size
            s_y = self.crop_size // 2
            s_x = self.crop_size // 2
            pad_h = (c_h - (H_orig % c_h)) % c_h
            pad_w = (c_w - (W_orig % c_w)) % c_w
            padded_img = cv2.copyMakeBorder(self.img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            H_pad, W_pad = padded_img.shape[:2]
            engine = GlobalConsensusInference(full_H=H_pad, full_W=W_pad, mask_thresh=0.5)
            tta_modes = [("Orig", None), ("FlipH", 1), ("FlipV", 0), ("FlipHV", -1)]
            coords = [(y, x) for y in range(0, H_pad - c_h + 1, s_y) for x in range(0, W_pad - c_w + 1, s_x)]
            total_steps = len(tta_modes) * len(coords)
            current_step = 0
            for tta_name, f_code in tta_modes:
                curr_img = cv2.flip(padded_img, f_code) if f_code is not None else padded_img
                for y, x in coords:
                    current_step += 1
                    if current_step % 5 == 0:
                        self.status.showMessage(f"AI Inference: {tta_name} sweep... ({current_step}/{total_steps})")
                        QApplication.processEvents()
                    gy, gx = get_global_coords(y, x, H_pad, W_pad, f_code, c_w)
                    h_y = np.hanning(c_h)
                    h_x = np.hanning(c_w)
                    if y == 0: 
                        h_y[:c_h // 2] = 1.0
                    if y + c_h >= H_pad: 
                        h_y[c_h // 2:] = 1.0
                    if x == 0: 
                        h_x[:c_w // 2] = 1.0
                    if x + c_w >= W_pad: 
                        h_x[c_w // 2:] = 1.0
                    h_win = (np.outer(h_y, h_x) * 0.9 + 0.1).astype(np.float32)
                    crop = curr_img[y:y + c_h, x:x + c_w]
                    crop_input = crop
                    img_t = self._preprocess(crop_input)
                    with torch.no_grad():
                        fields, masks, score_logits, _ = self.model(img_t)
                    valid_q = fields.get("query_valid", None)
                    if valid_q is not None:
                        q_mask = valid_q[0].detach().cpu().numpy()
                    else:
                        q_mask = (score_logits[0] > -10.0).detach().cpu().numpy()
                    if not np.any(q_mask):
                        continue
                    scores = torch.sigmoid(score_logits[0, q_mask])
                    selected_masks = masks[0, q_mask].unsqueeze(1)
                    masks_res = F.interpolate(selected_masks, size=(c_h, c_w), mode="bilinear", align_corners=False)[:, 0]
                    sem_res = F.interpolate(fields["semantic"], size=(c_h, c_w), mode="bilinear", align_corners=False)
                    engine.add_candidates(gy, gx, masks_res, scores, h_win, sem_res, f_code)
            self.status.showMessage("Resolving global consensus...")
            QApplication.processEvents()
            final_inst_padded = engine.resolve()
            final_inst = final_inst_padded[:H_orig, :W_orig]
            unique_ai_ids = np.unique(final_inst)
            unique_ai_ids = unique_ai_ids[unique_ai_ids > 0]
            instance_count = 0
            for ai_id in unique_ai_ids:
                ai_mask_bool = (final_inst == ai_id)
                total_ai_area = np.sum(ai_mask_bool)
                valid_space = (self.mask == 0)
                clean_mask = ai_mask_bool & valid_space
                clean_area = np.sum(clean_mask)
                if clean_area < 50 or (clean_area / total_ai_area) < 0.50:
                    continue
                next_id = self.get_next_free_id()
                if next_id >= 255: 
                    break
                if np.any(clean_mask):
                    self.mask[clean_mask] = next_id
                    self.ai_added_ids.add(next_id)
                    instance_count += 1
            self.set_active_id(self.get_next_free_id())
            self.status.showMessage(f"AI Complete: {instance_count} instances added.")
        except Exception as e:
            self.status.showMessage(f"Error: {str(e)}")
            print(f"Full Inference Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.update_display()

    def _init_colormaps(self):
        """Precomputes color palettes for human and machine annotated instances."""
        self.lut_manual = np.zeros((256, 3), dtype=np.uint8)
        self.lut_ai = np.zeros((256, 3), dtype=np.uint8)
        rng = np.random.RandomState(42)
        for i in range(1, 256):
            hue = rng.choice([rng.randint(0, 35), rng.randint(85, 179)])
            hsv = np.uint8([[[hue, rng.randint(150, 255), rng.randint(150, 255)]]])
            self.lut_manual[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
            hsv_ai = np.uint8([[[rng.choice([rng.randint(0, 10), rng.randint(170, 180)]), 
                                 rng.randint(150, 255), rng.randint(150, 255)]]])
            self.lut_ai[i] = cv2.cvtColor(hsv_ai, cv2.COLOR_HSV2RGB)[0, 0]

    def update_display(self):
        """Blends imagery, masks, void zones, and active drawings to refresh the raster view."""
        if self.img_rgb is None: return
        if self.brightness_slider.value() != 0:
            val = self.brightness_slider.value()
            base_canvas = cv2.add(self.img_rgb, np.array([val, val, val], dtype=np.float64))
        else:
            base_canvas = self.img_rgb.copy()
        if not self.hide_void:
            base_canvas[self.void_mask == 255] = [200, 200, 200]
        if not self.hide_mask and np.any(self.mask > 0):
            ai_mask_bool = np.isin(self.mask, list(self.ai_added_ids))
            manual_mask_bool = (self.mask > 0) & ~ai_mask_bool
            if np.any(manual_mask_bool):
                manual_colors = self.lut_manual[self.mask[manual_mask_bool]]
                base_canvas[manual_mask_bool] = cv2.addWeighted(
                    np.expand_dims(manual_colors, 0), 0.75, 
                    np.expand_dims(base_canvas[manual_mask_bool], 0), 0.25, 0
                )[0]
            if np.any(ai_mask_bool):
                ai_colors = self.lut_ai[self.mask[ai_mask_bool]]
                base_canvas[ai_mask_bool] = cv2.addWeighted(
                    np.expand_dims(ai_colors, 0), 0.5, 
                    np.expand_dims(base_canvas[ai_mask_bool], 0), 0.5, 0
                )[0]
        h, w, ch = base_canvas.shape
        q_img = QImage(base_canvas.data, w, h, ch * w, QImage.Format_RGB888)
        self.canvas.set_data(QPixmap.fromImage(q_img), self.zoom_factor)

    def get_next_free_id(self):
        """Scans the active instance mask to find the lowest available positive ID."""
        used_ids = set(np.unique(self.mask))
        for candidate in range(1, 255):
            if candidate not in used_ids:
                return candidate
        return 254

    def save_and_next(self):
        """Flushes composite segmentation masks to disk and advances image list indices."""
        fname = self.filenames[self.current_idx]
        out_img = os.path.join(self.source_img_dir, fname)
        out_mask = os.path.join(self.source_mask_dir, f"{os.path.splitext(fname)[0]}_mask.png")
        merged_mask = self.mask.copy()
        merged_mask[(self.mask == 0) & (self.void_mask == 255)] = 255
        cv2.imwrite(out_img, cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask, merged_mask.astype(np.uint8))
        self.status.showMessage(f"Saved: {fname}")
        if self.current_idx < len(self.filenames) - 1:
            self.file_list.setCurrentIndex(self.current_idx + 1)
        else:
            self.status.showMessage("Dataset complete. You can now Terminate Session.")

    def mark_as_done(self):
        """Prefixes completed image and mask files with done_ and drops them from working queues."""
        if not self.filenames: return
        fname = self.filenames[self.current_idx]
        new_fname = f"done_{fname}"
        out_img = os.path.join(self.source_img_dir, new_fname)
        out_mask = os.path.join(self.source_mask_dir, f"{os.path.splitext(new_fname)[0]}_mask.png")
        old_img = os.path.join(self.source_img_dir, fname)
        old_mask = os.path.join(self.source_mask_dir, f"{os.path.splitext(fname)[0]}_mask.png")
        merged_mask = self.mask.copy()
        merged_mask[(self.mask == 0) & (self.void_mask == 255)] = 255
        cv2.imwrite(out_img, cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask, merged_mask.astype(np.uint8))
        try:
            if os.path.exists(old_img):
                os.remove(old_img)
            if os.path.exists(old_mask):
                os.remove(old_mask)
        except Exception as e:
            print(f"Warning: Failed to delete old files after marking as done: {e}")
        data = self.file_dict.pop(fname)
        data['img'] = out_img
        data['mask'] = out_mask
        self.file_dict[new_fname] = data
        self.status.showMessage(f"Marked as done: {fname}")
        self.filenames.pop(self.current_idx)
        self.file_list.blockSignals(True)
        self.file_list.removeItem(self.current_idx)
        self.file_list.blockSignals(False)
        if self.filenames:
            if self.current_idx >= len(self.filenames):
                self.current_idx = len(self.filenames) - 1
            self.load_sample(self.current_idx)
        else:
            self.img_rgb = None
            self.mask = None
            self.canvas.set_data(QPixmap(), 1.0)
            self.status.showMessage("All samples are done! You can conclude the session.")


def run(device, config):
    """Entry point bootstrapping dataset exploration, neural model instances, and GUI execution."""
    root = str(Path(config.get("root")).resolve())
    file_dict = collect_files(root)
    new_root, s_img, s_mask = create_versioned_root(root)
    model = serialization.load_for_inference(config.get("weights"), device)
    crop_size = config.get("inference_window_size")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = LISNetLabeler(file_dict, s_img, s_mask, model, crop_size, device)
    gui.show()
    sys.exit(app.exec_())