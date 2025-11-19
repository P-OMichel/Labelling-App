import sys
import os
import json
import numpy as np

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class SpectrogramViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal & Spectrogram Viewer — 128 Hz")
        self.resize(1200, 800)

        # Fixed sampling frequency as requested
        self.fs = 128

        # Data holders
        self.signal = None  # 1D numpy array
        self.t_signal = None  # time array for signal

        # Spectrogram holders
        self.f = None
        self.t_spec = None
        self.Sxx = None  # power (in dB)

        # Overlay curve holders
        self.base_overlay = None  # base random curve [0, 1]
        self.overlay_curve_item = None
        self.threshold_line_item = None

        # Segment highlighting holders
        self.segment_items = []

        # Labelling holders
        self.label_classes = [ "IES (large)", "IES (BS)", "hf and IES", "IES (other)", "alpha-supp", "eye artifact", "lf artifacts", "hf artifact", "all freqs artifact", "large artifacts", "ground check", "shallow signal", "awake hf signal", "OK", "heavy without suppressions", "high gamma signal"]
        self.class_colors = {
            "IES (large)": (255, 165, 0, 80),   # orange semi-transparent
            "IES (BS)": (255, 0, 0, 80),  # red
            "hf and IES" : (0, 0, 255, 80),  # blue
            "IES (other)": (0, 128, 255, 80),   # blue-ish
            "alpha-supp": (128, 0, 255, 80),   # purple
            "eye artifact": (0, 200, 0, 80),     # green
            "lf artifact": (0, 200, 200, 80),     # green
            "hf artifact": (0, 255, 200, 80),     # green
            "all freqs artifact": (0, 255, 150, 80),     # green
            "large artifacts": (0, 255, 100, 80),     # green
            "ground check": (0, 0, 0, 80),     # green
            "shallow signal": (100, 100, 0, 80),     # green
            "awake hf signal": (150, 150, 0, 80),
            "OK": (100, 150, 0, 80),
            "heavy without suppressions" : (125, 175, 0, 80),
            "high gamma signal" : (150, 200, 0, 80)
        }

        self.labels = []   # list of dicts: {id, start, end, cls}
        self.label_items = {}  # id -> LinearRegionItem
        self.label_text_items = {}  # id -> TextItem
        self._label_id_seq = 1
        self._label_active = False
        self.current_file_path = None

        # Build UI
        self._build_ui()

    # ---------------- UI -----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton("Load .npy…")
        btn_load.clicked.connect(self.load_npy)
        controls.addWidget(btn_load)

        # Zoom-by-rectangle
        self.select_btn = QtWidgets.QPushButton("Select Range")
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self.toggle_select_mode)
        controls.addWidget(self.select_btn)

        # Home/reset zoom button
        self.home_btn = QtWidgets.QPushButton("Home")
        self.home_btn.clicked.connect(self.reset_zoom)
        controls.addWidget(self.home_btn)

        # ---- Labelling controls ----
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Class:"))
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.addItems(self.label_classes)
        controls.addWidget(self.class_combo)

        self.label_btn = QtWidgets.QPushButton("Label Mode")
        self.label_btn.setCheckable(True)
        self.label_btn.toggled.connect(self.toggle_label_mode)
        controls.addWidget(self.label_btn)

        self.save_labels_btn = QtWidgets.QPushButton("Save Labels")
        self.save_labels_btn.clicked.connect(self.save_labels)
        controls.addWidget(self.save_labels_btn)

        self.delete_label_btn = QtWidgets.QPushButton("Delete Selected")
        self.delete_label_btn.clicked.connect(self.delete_selected_label)
        controls.addWidget(self.delete_label_btn)

        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 50)
        self.threshold_spin.setSingleStep(1)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(10)
        self.threshold_spin.setFixedWidth(100)
        self.threshold_spin.valueChanged.connect(self.update_overlay)
        controls.addWidget(self.threshold_spin)

        controls.addStretch(1)
        vbox.addLayout(controls)

        # Plots
        # Shared-axis: link X of signal and spectrogram
        self.plot_signal = pg.PlotWidget()
        self.plot_signal.setLabel('bottom', 'Time', units='s')
        self.plot_signal.setLabel('left', 'Amplitude')
        self.plot_signal.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_signal, stretch=3)
        # keep label captions positioned when zoom/pan
        self.plot_signal.sigRangeChanged.connect(self._on_signal_view_changed)

        # Spectrogram plot (use ImageItem inside a PlotWidget to overlay curves easily)
        self.plot_spec = pg.PlotWidget()
        # Link X axes
        self.plot_spec.setXLink(self.plot_signal)
        self.plot_spec.setLabel('bottom', 'Time', units='s')
        self.plot_spec.setLabel('left', 'Frequency', units='Hz')
        self.plot_spec.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_spec, stretch=4)

        # Image item for spectrogram
        self.img_item = pg.ImageItem()
        self.plot_spec.addItem(self.img_item)
        self.img_item.setOpts(axisOrder='row-major')
        try:
            self.img_item.setAutoDownsample(True)
        except Exception:
            pass

        # Color map
        cmap = pg.colormap.get('turbo')
        self.img_item.setLookupTable(cmap.getLookupTable())

        # Overlay curve and threshold line
        self.overlay_curve_item = pg.PlotDataItem(pen=pg.mkPen(width=2))
        self.plot_spec.addItem(self.overlay_curve_item)

        self.threshold_line_item = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self.plot_spec.addItem(self.threshold_line_item)

        self.setCentralWidget(central)

        # Labels list dock
        self._build_labels_dock()

        # Rubber-band for rectangle selection and labelling on the top plot
        self._install_rubber_band()

    # --------------- Rubber-band selection ---------------
    def _install_rubber_band(self):
        # Use the PlotWidget's viewport for proper coordinate mapping
        self._rb_origin = None
        self._rb_active = False
        self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.plot_signal.viewport())
        # Listen to mouse events only on the top plot when selection/label mode is on
        self.plot_signal.viewport().installEventFilter(self)

    def toggle_select_mode(self, enabled: bool):
        self._rb_active = enabled
        if enabled:
            # ensure label mode is off
            self.label_btn.setChecked(False)
            self.plot_signal.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.plot_signal.viewport().unsetCursor()
            self.rubber_band.hide()
            self._rb_origin = None

    def toggle_label_mode(self, enabled: bool):
        self._label_active = enabled
        if enabled:
            # ensure select mode is off
            self.select_btn.setChecked(False)
            self.plot_signal.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.plot_signal.viewport().unsetCursor()
            self.rubber_band.hide()
            self._rb_origin = None

    def eventFilter(self, obj, event):
        # Only handle events for the top plot's viewport when selection or label mode is on
        if obj is self.plot_signal.viewport() and (self._rb_active or self._label_active):
            if event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._rb_origin = event.position().toPoint()
                self.rubber_band.setGeometry(QtCore.QRect(self._rb_origin, QtCore.QSize()))
                self.rubber_band.show()
                return True
            elif event.type() == QtCore.QEvent.Type.MouseMove and self._rb_origin is not None:
                current = event.position().toPoint()
                rect = QtCore.QRect(self._rb_origin, current).normalized()
                self.rubber_band.setGeometry(rect)
                return True
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self._rb_origin is not None:
                    current = event.position().toPoint()
                    rect = QtCore.QRect(self._rb_origin, current).normalized()
                    self.rubber_band.hide()
                    if self._label_active:
                        self._apply_rect_label(rect)
                        # keep label mode on for multiple labels
                    else:
                        self._apply_rect_zoom(rect)
                        # turn off selection mode after one use
                        self.select_btn.setChecked(False)
                return True
        return super().eventFilter(obj, event)

    def reset_zoom(self):
        # Reset to full data range
        if self.t_signal is not None and len(self.t_signal) > 1:
            self.plot_signal.enableAutoRange()
            self.plot_spec.enableAutoRange()

    def _apply_rect_zoom(self, rect: QtCore.QRect):
        if rect.width() < 4 or rect.height() < 4:
            return  # ignore tiny drags
        # Map viewport coords -> scene -> data coords
        scene_tl = self.plot_signal.mapToScene(rect.topLeft())
        scene_br = self.plot_signal.mapToScene(rect.bottomRight())
        vb = self.plot_signal.plotItem.vb
        data_tl = vb.mapSceneToView(scene_tl)
        data_br = vb.mapSceneToView(scene_br)
        x0, x1 = sorted([data_tl.x(), data_br.x()])
        self.plot_signal.setXRange(x0, x1, padding=0)
        self.plot_spec.setXRange(x0, x1, padding=0)

    def _apply_rect_label(self, rect: QtCore.QRect):
        if rect.width() < 4:
            return
        scene_tl = self.plot_signal.mapToScene(rect.topLeft())
        scene_br = self.plot_signal.mapToScene(rect.bottomRight())
        vb = self.plot_signal.plotItem.vb
        data_tl = vb.mapSceneToView(scene_tl)
        data_br = vb.mapSceneToView(scene_br)
        start, end = sorted([data_tl.x(), data_br.x()])
        cls = self.class_combo.currentText()
        self._create_label(start, end, cls)

    # --------------- Labels dock & helpers ---------------
    def _on_signal_view_changed(self, *args, **kwargs):
        # keep all label captions inside view when panning/zooming
        for lab in self.labels:
            self._place_label_text(lab["id"]) 

    def _ensure_label_text(self, label_id: int, cls: str):
        # create if not exists, then place
        if label_id not in self.label_text_items:
            # Text with small background for readability
            txt = pg.TextItem(anchor=(0.5, 1.0))
            # style: id | class
            txt.setHtml(f"<div style='background: rgba(255,255,255,0.7); padding:2px 4px; border-radius:4px; font-size:11px;'># {label_id} | {cls}</div>")
            self.label_text_items[label_id] = txt
            self.plot_signal.addItem(txt)
        else:
            # update class text if changed
            txt = self.label_text_items[label_id]
            txt.setHtml(f"<div style='background: rgba(255,255,255,0.7); padding:2px 4px; border-radius:4px; font-size:11px;'># {label_id} | {cls}</div>")
        self._place_label_text(label_id)

    def _place_label_text(self, label_id: int):
        region = self.label_items.get(label_id)
        text = self.label_text_items.get(label_id)
        if region is None or text is None:
            return
        start, end = region.getRegion()
        x = 0.5 * (start + end)
        # place near top of current view
        vb = self.plot_signal.getViewBox()
        if vb is None:
            return
        yr = vb.viewRange()[1]
        y = yr[1] - 0.06 * (yr[1] - yr[0])
        text.setPos(x, y)

    def _build_labels_dock(self):
        self.labels_dock = QtWidgets.QDockWidget("Labels", self)
        self.labels_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        dock_widget = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(dock_widget)
        self.labels_list = QtWidgets.QListWidget()
        self.labels_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        v.addWidget(self.labels_list)
        self.labels_dock.setWidget(dock_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.labels_dock)

    def _create_label(self, start: float, end: float, cls: str, label_id: int = None):
        # clamp to data range
        if self.t_signal is None:
            return
        x_min, x_max = self.t_signal[0], self.t_signal[-1]
        start = max(x_min, float(start))
        end = min(x_max, float(end))
        if end - start <= 0.0:
            return
        if label_id is None:
            label_id = self._label_id_seq
        self._label_id_seq = max(self._label_id_seq + 1, label_id + 1)
        # visual region
        color = self.class_colors.get(cls, (0, 200, 0, 80))
        region = pg.LinearRegionItem(values=(start, end), orientation='vertical')
        region.setBrush(pg.mkBrush(*color))
        region.setZValue(10)
        # allow user adjustment of left/right borders
        region.lines[0].setMovable(True)
        region.lines[1].setMovable(True)
        region.sigRegionChanged.connect(lambda r=region, lid=label_id: self._label_region_changed(lid, r))
        self.plot_signal.addItem(region)
        self.labels.append({"id": label_id, "start": float(start), "end": float(end), "class": cls})
        self.label_items[label_id] = region
        self._append_label_to_list(label_id, start, end, cls)
        # create/update caption
        self._ensure_label_text(label_id, cls)

    def _append_label_to_list(self, label_id: int, start: float, end: float, cls: str):
        item = QtWidgets.QListWidgetItem(f"#{label_id} | {cls} | {start:.3f}s–{end:.3f}s")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, label_id)
        self.labels_list.addItem(item)

    def _label_region_changed(self, label_id, region):
        # Update stored label values when user adjusts borders
        start, end = region.getRegion()
        cls = None
        for lab in self.labels:
            if lab["id"] == label_id:
                lab["start"] = float(start)
                lab["end"] = float(end)
                cls = lab["class"]
                break
        # update list entry
        for i in range(self.labels_list.count()):
            item = self.labels_list.item(i)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == label_id:
                item.setText(f"#{label_id} | {cls} | {start:.3f}s–{end:.3f}s")
                break
        # reposition caption
        self._place_label_text(label_id)
        # update list entry
        for i in range(self.labels_list.count()):
            item = self.labels_list.item(i)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == label_id:
                cls = lab["class"]
                item.setText(f"#{label_id} | {cls} | {start:.3f}s–{end:.3f}s")
                break

    def delete_selected_label(self):
        item = self.labels_list.currentItem()
        if not item:
            return
        label_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        # remove from list widget
        row = self.labels_list.row(item)
        self.labels_list.takeItem(row)
        # remove visual
        region = self.label_items.pop(label_id, None)
        if region is not None:
            self.plot_signal.removeItem(region)
        # remove caption
        text = self.label_text_items.pop(label_id, None)
        if text is not None:
            self.plot_signal.removeItem(text)
        # remove from data
        self.labels = [d for d in self.labels if d["id"] != label_id]

    def save_labels(self):
        if not self.labels:
            QtWidgets.QMessageBox.information(self, "Save Labels", "No labels to save.")
            return
        # default path next to current file
        default_path = None
        if self.current_file_path:
            base, _ = os.path.splitext(self.current_file_path)
            default_path = base + "_labels.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save labels as JSON", default_path or os.getcwd(), "JSON (*.json)")
        if not path:
            return
        payload = {
            "fs": self.fs,
            "n": int(self.signal.size) if self.signal is not None else 0,
            "labels": self.labels,
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save labels:{e}")
        else:
            QtWidgets.QMessageBox.information(self, "Save Labels", f"Saved {len(self.labels)} labels to:{path}")

    # --------------- Labels persistence ---------------
    def clear_all_labels(self):
        # remove visuals
        for rid, item in list(self.label_items.items()):
            try:
                self.plot_signal.removeItem(item)
            except Exception:
                pass
        for rid, txt in list(self.label_text_items.items()):
            try:
                self.plot_signal.removeItem(txt)
            except Exception:
                pass
        self.label_text_items.clear()
        self.label_items.clear()
        self.labels_list.clear()
        self.labels = []
        self._label_id_seq = 1

    def load_labels_json(self, json_path: str):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Labels", f"Could not read labels file:{e}")
            return
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            return
        # Recreate labels and keep their IDs
        for lab in labels:
            try:
                lid = int(lab.get("id"))
                start = float(lab.get("start"))
                end = float(lab.get("end"))
                cls = str(lab.get("class", "Other"))
            except Exception:
                continue
            self._create_label(start, end, cls, label_id=lid)

    # --------------- File I/O ---------------
    def load_npy(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .npy file", os.getcwd(), "NumPy array (*.npy)")
        if not path:
            return
        try:
            arr = np.load(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Could not load file:{e}")
            return

        if arr.ndim != 1:
            QtWidgets.QMessageBox.warning(self, "Invalid Shape", f"Expected a 1D array. Got shape {arr.shape}.")
            return

        # Reset labels for new file and set file path
        self.clear_all_labels()
        self.current_file_path = path
        self.signal = arr.astype(float)
        n = self.signal.size
        self.t_signal = np.arange(n) / self.fs

        # Update time-domain plot
        self.plot_signal.clear()
        # main signal curve with fast downsampling in view
        main_curve = self.plot_signal.plot(self.t_signal, self.signal, pen=pg.mkPen(width=1))
        try:
            # available in recent pyqtgraph
            main_curve.setDownsampling(auto=True, method='peak')
            main_curve.setClipToView(True)
        except Exception:
            pass
        self.signal_curve = main_curve
        self.plot_signal.enableAutoRange()

        # --- Random mask-based highlighting (binary mask) ---
        from Functions.suppressions import detect_supp_threshold
        mask_supp = detect_supp_threshold(self.signal, self.fs, 15)
        masked_signal = self.signal.copy()
        masked_signal[mask_supp == 0] = np.nan
        red_curve = self.plot_signal.plot(self.t_signal, masked_signal, pen=pg.mkPen(QtGui.QColor('red'), width=2), connect='finite')
        try:
            red_curve.setDownsampling(auto=True, method='peak')
            red_curve.setClipToView(True)
        except Exception:
            pass

        # Compute spectrogram and overlay
        self.recompute_spectrogram()
        self.create_or_update_base_overlay()
        self.update_overlay()

        # Auto-load labels JSON if present next to the .npy
        base, _ = os.path.splitext(self.current_file_path)
        labels_json = base + "_labels.json"
        if os.path.exists(labels_json):
            self.load_labels_json(labels_json)

    # --------------- Spectrogram ---------------
    def recompute_spectrogram(self):
        if self.signal is None:
            return
        #from scipy.signal import spectrogram
        from Functions.time_frequency import spectrogram

        t, f, Sxx = spectrogram(self.signal, fs=self.fs, nperseg_factor=1, nfft_factor=2)

        self.f, self.t_spec, self.Sxx = f, t, Sxx
        self._update_spectrogram_image()
        self.update_overlay()

    def _update_spectrogram_image(self):
        if self.Sxx is None:
            return
        # Sxx shape: (n_freqs, n_times)
        # Display in dB with reasonable dynamic range
        Sxx = self.Sxx.copy()
        Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-20))

        # Set image
        # Use percentile-based levels for speed and robustness
        finite = np.isfinite(Sxx_db)
        if np.any(finite):
            pdown, pup = np.percentile(Sxx_db[finite], [50, 95])
            self.img_item.setImage(Sxx_db, autoLevels=False, levels=(float(pdown)-1, float(pup)+1))
        else:
            self.img_item.setImage(Sxx_db, autoLevels=True)

        # Map spectrogram array coordinates to time/frequency axes
        # ImageItem expects axes: x=time, y=frequency — set appropriate transform
        # Define image rectangle: left=min(t), bottom=min(f), width=dt*ncols, height=df*nrows
        if len(self.t_spec) > 1:
            dt = self.t_spec[1] - self.t_spec[0]
        else:
            dt = 1.0 / self.fs
        if len(self.f) > 1:
            df = self.f[1] - self.f[0]
        else:
            df = self.fs / 2.0
        left = self.t_spec[0] if len(self.t_spec) else 0.0
        bottom = self.f[0] if len(self.f) else 0.0
        width = dt * Sxx_db.shape[1]
        height = df * Sxx_db.shape[0]
        self.img_item.setRect(QtCore.QRectF(left, bottom, width, height))

        # Update axes ranges neatly
        self.plot_spec.setXRange(left, left + width, padding=0)
        self.plot_spec.setYRange(bottom, bottom + height, padding=0)

    # --------------- Overlay curve ---------------
    def create_or_update_base_overlay(self):
        if self.t_spec is None or len(self.t_spec) == 0:
            return
        # Base random curve in [0, 1], smoothed
        rng = np.random.default_rng()
        base = rng.random(len(self.t_spec))
        # Smooth with simple moving average
        k = max(3, int(round(0.05 * len(base))))
        kernel = np.ones(k) / k
        base_sm = np.convolve(base, kernel, mode='same')
        base_sm = (base_sm - base_sm.min()) / max(1e-12, (base_sm.max() - base_sm.min()))
        self.base_overlay = base_sm

    def update_overlay(self):
        if self.base_overlay is None or self.f is None or self.t_spec is None:
            return
        from Functions.edge_frequency import edge_frequencies_significant_value

        thr = float(self.threshold_spin.value())  

        ef = edge_frequencies_significant_value(self.Sxx, self.f, max_val = 50, threshold= thr)[0]

        # Update curve plot
        self.overlay_curve_item.setData(self.t_spec, ef)
        
    # --------------- Utils ---------------
    def keyPressEvent(self, event):
        # Quick refresh/regenerate overlay with 'R'
        if event.key() == QtCore.Qt.Key.Key_R:
            self.create_or_update_base_overlay()
            self.update_overlay()
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Autosave labels to JSON next to the current file on exit."""
        try:
            if self.current_file_path and self.labels:
                base, _ = os.path.splitext(self.current_file_path)
                out_path = base + "_labels.json"
                payload = {
                    "fs": self.fs,
                    "n": int(self.signal.size) if self.signal is not None else 0,
                    "labels": self.labels,
                }
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
        except Exception:
            # don't block closing on autosave errors
            pass
        finally:
            super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = SpectrogramViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()






