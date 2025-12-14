# hw3/gui.py（示例片段）
from tkinter import Tk, Frame, Label, Button, Scale, Entry, LEFT, TOP, X, HORIZONTAL, Radiobutton, Canvas, filedialog, StringVar, IntVar
from PIL import Image, ImageTk, ImageDraw
import math
import random
from oop_logic import NegativeFilter, GrayFilter, PaletteQuantizer, MedianFilter, LaplacianEdgeFilter, LaplacianSharpenFilter


class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter GUI")
        self.original_img = None
        self.undo_stack = []
        self.preview_img = None
        self._draw = None
        self._doodle_active = False
        self.last_xy = None
        self.scatter_density = 0.25
        self.brush_size = 8
        self.brush_color = (0, 0, 0)
        self.brush_type_var = IntVar(value=0)
        self.r_var = StringVar(value="0")
        self.g_var = StringVar(value="0")
        self.b_var = StringVar(value="0")

        self.toolbar_frame = Frame(root)
        self.toolbar_frame.pack(side=TOP, fill=X, pady=5)

        self.top_frame = Frame(root)
        self.top_frame.pack(side=TOP, fill=X, pady=5)

        self.palette_slider = Scale(self.top_frame, from_=2, to=256, orient=HORIZONTAL,
                                    label="colors", length=500,
                                    command=self.on_palette_slider)
        self.palette_entry_var = StringVar(value="16")
        self.palette_entry = Entry(self.top_frame, width=5, textvariable=self.palette_entry_var)
        self.palette_entry.bind("<Return>", self.on_palette_entry)
        self.palette_frame_visible = False
        self.palette_slider.set(16)

        self.median_slider = Scale(self.top_frame, from_=1, to=15, orient=HORIZONTAL,
                                    label="median size", length=500,
                                    command=self.on_median_slider)
        self.median_frame_visible = False

        self.laplacian_kernel_var = IntVar(value=4)
        self.laplacian_kernel_r4 = Radiobutton(self.top_frame, text="kernel 4", variable=self.laplacian_kernel_var, value=4, command=self.on_laplacian_kernel)
        self.laplacian_kernel_r8 = Radiobutton(self.top_frame, text="kernel 8", variable=self.laplacian_kernel_var, value=8, command=self.on_laplacian_kernel)
        self.laplacian_frame_visible = False

        self.sharpen_strength_slider = Scale(self.top_frame, from_=0, to=100, resolution=1, orient=HORIZONTAL,
                                             label="銳利強度(0-100)", length=500,
                                             command=self.on_sharpen_strength_slider)
        self.sharpen_frame_visible = False

        self.apply_button = Button(self.top_frame, text="apply", command=self.on_apply_button)
        self.apply_visible = False

        self.doodle_size_slider = Scale(self.top_frame, from_=1, to=64, orient=HORIZONTAL, label="brush size", length=300, command=self.on_doodle_size_change)
        self.doodle_r_pixel = Radiobutton(self.top_frame, text="pixel", variable=self.brush_type_var, value=0)
        self.doodle_r_scatter = Radiobutton(self.top_frame, text="scatter", variable=self.brush_type_var, value=1)
        self.color_swatch = Canvas(self.top_frame, width=40, height=24, highlightthickness=1)
        self.rgb_row = Frame(self.top_frame)
        self.rgb_r_label = Label(self.rgb_row, text="R")
        self.rgb_r_entry = Entry(self.rgb_row, width=4, textvariable=self.r_var)
        self.rgb_g_label = Label(self.rgb_row, text="G")
        self.rgb_g_entry = Entry(self.rgb_row, width=4, textvariable=self.g_var)
        self.rgb_b_label = Label(self.rgb_row, text="B")
        self.rgb_b_entry = Entry(self.rgb_row, width=4, textvariable=self.b_var)
        self.r_var.trace_add("write", self.on_rgb_change)
        self.g_var.trace_add("write", self.on_rgb_change)
        self.b_var.trace_add("write", self.on_rgb_change)
        self._doodle_controls_visible = False
        self.update_swatch()

        self.left_frame = Frame(root)
        self.left_frame.pack(side=LEFT, fill="y", padx=6, pady=6)

        Button(self.toolbar_frame, text="Open…", command=self.open_image).pack(side=LEFT, padx=6)
        Button(self.toolbar_frame, text="Save…", command=self.save_image).pack(side=LEFT, padx=6)
        Button(self.toolbar_frame, text="Revert", command=self.show_original).pack(side=LEFT, padx=6)
        Button(self.toolbar_frame, text="Undo", command=self.undo).pack(side=LEFT, padx=6)

        Button(self.left_frame, text="Negative", command=lambda: self.apply_filter(NegativeFilter())).pack(pady=6)
        Button(self.left_frame, text="Grayscale", command=lambda: self.apply_filter(GrayFilter())).pack(pady=6)
        Button(self.left_frame, text="Palette", command=self.apply_palette_mode).pack(pady=6)
        Button(self.left_frame, text="Median filter", command=self.apply_median_mode).pack(pady=6)
        Button(self.left_frame, text="Laplacian edge", command=self.apply_laplacian_mode).pack(pady=6)
        Button(self.left_frame, text="Sharpen filter", command=self.apply_sharpen_mode).pack(pady=6)
        Button(self.left_frame, text="Doodle", command=self.apply_doodle_mode).pack(pady=6)

        # 右侧：图像展示
        self.label = Canvas(root, width=500, height=500, highlightthickness=0)
        self.label.pack(side=LEFT, padx=5, pady=5)
        self.label.bind("<ButtonPress-1>", self.on_mouse_down)
        self.label.bind("<B1-Motion>", self.on_mouse_drag)
        self.label.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.roi = None
        self._selection_rect_id = None
        self._selection_active = False
        self._drag_start = None

        # 防抖定时器ID
        self._palette_after_id = None
        self._median_after_id = None
        self._laplacian_after_id = None
        self._sharpen_after_id = None

    def show_image(self, img):
        if img is None:
            return
        w, h = img.size
        scale = min(500 / w, 500 / h)
        dw, dh = int(w * scale), int(h * scale)
        ox = (500 - dw) // 2
        oy = (500 - dh) // 2
        img_resized = img.resize((dw, dh))
        tk_img = ImageTk.PhotoImage(img_resized)
        self.label.delete("all")
        self.label.create_image(ox, oy, image=tk_img, anchor='nw')
        self.label.image = tk_img
        self.display_scale = scale
        self.display_offset_x = ox
        self.display_offset_y = oy
        self._disp_width = dw
        self._disp_height = dh
        if self.roi is not None:
            self.draw_roi_rect()

    def open_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.original_img = Image.open(path).convert("RGB")
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.show_image(self.original_img)
        self.preview_img = None
        self.hide_apply_button()
        self.undo_stack.clear()
        self._doodle_active = False

    def hide_all_controls(self):
        try:
            self.palette_slider.pack_forget()
            self.palette_entry.pack_forget()
            self.median_slider.pack_forget()
            self.laplacian_kernel_r4.pack_forget()
            self.laplacian_kernel_r8.pack_forget()
            self.sharpen_strength_slider.pack_forget()
            self.apply_button.pack_forget()
            if self._doodle_controls_visible:
                self.doodle_size_slider.pack_forget()
                self.doodle_r_pixel.pack_forget()
                self.doodle_r_scatter.pack_forget()
                self.color_swatch.pack_forget()
                self.rgb_row.pack_forget()
        except Exception:
            pass
        self.palette_frame_visible = False
        self.median_frame_visible = False
        self.laplacian_frame_visible = False
        self.sharpen_frame_visible = False
        self.apply_visible = False
        self._selection_active = False
        self._doodle_active = False
        self.clear_selection_overlay()
        self.roi = None
        self._doodle_controls_visible = False

    def apply_filter(self, filter_obj):
        if self.original_img is None:
            return
        self.cancel_all_scheduled()
        self.hide_all_controls()
        # 通用滤镜入口
        out = filter_obj.apply(self.original_img)
        self.preview_img = out
        self.show_apply_button()
        self.show_image(self.preview_img)

    # 调色盘模式
    def apply_palette_mode(self):
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.show_palette_controls()
        self.update_palette_immediate()

    def show_palette_controls(self):
        if not self.palette_frame_visible:
            self.palette_slider.pack(side=LEFT, padx=5)
            self.palette_entry.pack(side=LEFT, padx=5)
            self.palette_frame_visible = True

    def hide_palette_controls(self):
        if self.palette_frame_visible:
            self.palette_slider.pack_forget()
            self.palette_entry.pack_forget()
            self.palette_frame_visible = False

    def on_palette_slider(self, _):
        # 将 slider 值同步到 Entry
        value = self.palette_slider.get()
        self.palette_entry_var.set(str(value))
        self.schedule_palette_update()

    def on_palette_entry(self, event=None):
        val = self.palette_entry_var.get()
        if val.isdigit():
            n = max(2, min(int(val), 256))
            self.palette_slider.set(n)
            self.schedule_palette_update()

    def schedule_palette_update(self):
        # 防抖：避免拖动时频繁计算
        if self._palette_after_id:
            self.root.after_cancel(self._palette_after_id)
        self._palette_after_id = self.root.after(200, self.update_palette_immediate)

    def update_palette_immediate(self):
        if self.original_img is None:
            return
        n = self.palette_slider.get()
        # 如果不想引入 scikit-learn，可设置 fallback=True
        pal_img = PaletteQuantizer(n_colors=n, fallback=True).apply(self.original_img)
        self.preview_img = pal_img
        self.show_apply_button()
        self.show_image(self.preview_img)

    def apply_median_mode(self):
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.show_median_controls()
        self._selection_active = True
        self.update_median_immediate()

    def show_median_controls(self):
        if not self.median_frame_visible:
            self.median_slider.pack(side=LEFT, padx=5)
            self.median_frame_visible = True

    def hide_median_controls(self):
        if self.median_frame_visible:
            self.median_slider.pack_forget()
            self.median_frame_visible = False
        self._selection_active = False

    def on_median_slider(self, _):
        self.schedule_median_update()

    def schedule_median_update(self):
        if self._median_after_id:
            self.root.after_cancel(self._median_after_id)
        self._median_after_id = self.root.after(200, self.update_median_immediate)

    def update_median_immediate(self):
        if self.original_img is None:
            return
        size = self.median_slider.get()
        out = MedianFilter().apply(self.original_img, size=size, roi=self.roi)
        self.preview_img = out
        self.show_apply_button()
        self.show_image(self.preview_img)

    def clamp255(self, v):
        try:
            iv = int(v)
        except Exception:
            iv = 0
        return max(0, min(255, iv))

    def on_rgb_change(self, *args):
        r = self.clamp255(self.r_var.get())
        g = self.clamp255(self.g_var.get())
        b = self.clamp255(self.b_var.get())
        self.brush_color = (r, g, b)
        self.update_swatch()

    def update_swatch(self):
        hex_color = f"#{self.brush_color[0]:02x}{self.brush_color[1]:02x}{self.brush_color[2]:02x}"
        try:
            self.color_swatch.delete("all")
            self.color_swatch.create_rectangle(0, 0, 40, 24, fill=hex_color, outline="black")
        except Exception:
            pass

    def on_doodle_size_change(self, _):
        self.brush_size = int(self.doodle_size_slider.get())

    def show_doodle_controls(self):
        if not self._doodle_controls_visible:
            self.doodle_size_slider.set(self.brush_size)
            self.doodle_size_slider.pack(side=LEFT, padx=5)
            self.doodle_r_pixel.pack(side=LEFT, padx=5)
            self.doodle_r_scatter.pack(side=LEFT, padx=5)
            self.rgb_row.pack(side=LEFT, padx=5)
            self.rgb_r_label.pack(side=LEFT)
            self.rgb_r_entry.pack(side=LEFT, padx=(2, 8))
            self.rgb_g_label.pack(side=LEFT)
            self.rgb_g_entry.pack(side=LEFT, padx=(2, 8))
            self.rgb_b_label.pack(side=LEFT)
            self.rgb_b_entry.pack(side=LEFT, padx=(2, 8))
            self.color_swatch.pack(side=LEFT, padx=5)
            self._doodle_controls_visible = True

    # hide_doodle_controls is unnecessary because hide_all_controls handles packing state

    def apply_doodle_mode(self):
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.show_doodle_controls()
        self._doodle_active = True
        if self.original_img is not None:
            self.preview_img = self.original_img.copy()
            self._draw = ImageDraw.Draw(self.preview_img)
            self.show_image(self.preview_img)
            self.show_apply_button()

    def clear_selection_overlay(self):
        if self._selection_rect_id:
            self.label.delete(self._selection_rect_id)
            self._selection_rect_id = None

    def canvas_to_image(self, x, y):
        if self.original_img is None:
            return 0, 0
        w, h = self.original_img.size
        ix = int((x - self.display_offset_x) / self.display_scale)
        iy = int((y - self.display_offset_y) / self.display_scale)
        ix = max(0, min(w, ix))
        iy = max(0, min(h, iy))
        return ix, iy

    def inside_image_area(self, x, y):
        return (self.display_offset_x <= x < self.display_offset_x + getattr(self, "_disp_width", 0)) and (self.display_offset_y <= y < self.display_offset_y + getattr(self, "_disp_height", 0))

    def draw_roi_rect(self):
        if self.roi is None:
            return
        x0, y0, x1, y1 = self.roi
        s = self.display_scale
        ox = self.display_offset_x
        oy = self.display_offset_y
        cx0 = ox + int(x0 * s)
        cy0 = oy + int(y0 * s)
        cx1 = ox + int(x1 * s)
        cy1 = oy + int(y1 * s)
        if self._selection_rect_id:
            self.label.delete(self._selection_rect_id)
        self._selection_rect_id = self.label.create_rectangle(cx0, cy0, cx1, cy1, outline="red", width=2)

    def on_mouse_down(self, event):
        if self._doodle_active:
            if self.preview_img is None and self.original_img is not None:
                self.preview_img = self.original_img.copy()
                self._draw = ImageDraw.Draw(self.preview_img)
            if not self.inside_image_area(event.x, event.y) or self.preview_img is None:
                return
            ix, iy = self.canvas_to_image(event.x, event.y)
            if self.brush_type_var.get() == 0:
                self.apply_stamp(ix, iy)
            else:
                self.apply_scatter(ix, iy)
            self.last_xy = (ix, iy)
            self.show_image(self.preview_img)
            self.show_apply_button()
            return
        if not self._selection_active or self.original_img is None:
            return
        self._drag_start = (event.x, event.y)
        if self._selection_rect_id:
            self.label.delete(self._selection_rect_id)
            self._selection_rect_id = None
        self._selection_rect_id = self.label.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def on_mouse_drag(self, event):
        if self._doodle_active:
            if self.preview_img is None or self.last_xy is None:
                return
            if not self.inside_image_area(event.x, event.y):
                return
            nx, ny = self.canvas_to_image(event.x, event.y)
            lx, ly = self.last_xy
            if self.brush_type_var.get() == 0:
                self.apply_path(lx, ly, nx, ny)
            else:
                self.apply_scatter_path(lx, ly, nx, ny)
            self.last_xy = (nx, ny)
            self.show_image(self.preview_img)
            return
        if not self._selection_active or self._drag_start is None:
            return
        sx, sy = self._drag_start
        self.label.coords(self._selection_rect_id, sx, sy, event.x, event.y)

    def on_mouse_up(self, event):
        if self._doodle_active:
            self.last_xy = None
            return
        if not self._selection_active or self._drag_start is None:
            return
        sx, sy = self._drag_start
        ex, ey = event.x, event.y
        self._drag_start = None
        x0, y0 = self.canvas_to_image(sx, sy)
        x1, y1 = self.canvas_to_image(ex, ey)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        self.roi = (x0, y0, x1, y1)
        self.draw_roi_rect()
        self.update_median_immediate()

    def apply_laplacian_mode(self):
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.show_laplacian_controls()
        self.update_laplacian_immediate()

    def show_laplacian_controls(self):
        if not self.laplacian_frame_visible:
            self.laplacian_kernel_r4.pack(side=LEFT, padx=5)
            self.laplacian_kernel_r8.pack(side=LEFT, padx=5)
            self.laplacian_frame_visible = True

    def hide_laplacian_controls(self):
        if self.laplacian_frame_visible:
            self.laplacian_kernel_r4.pack_forget()
            self.laplacian_kernel_r8.pack_forget()
            self.laplacian_frame_visible = False

    def on_laplacian_kernel(self):
        self.schedule_laplacian_update()

    def schedule_laplacian_update(self):
        if self._laplacian_after_id:
            self.root.after_cancel(self._laplacian_after_id)
        self._laplacian_after_id = self.root.after(200, self.update_laplacian_immediate)

    def update_laplacian_immediate(self):
        if self.original_img is None:
            return
        k = "4" if self.laplacian_kernel_var.get() == 4 else "8"
        out = LaplacianEdgeFilter(kernel=k).apply(self.original_img)
        self.preview_img = out
        self.show_apply_button()
        self.show_image(self.preview_img)

    def apply_sharpen_mode(self):
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.show_laplacian_controls()
        self.show_sharpen_controls()
        self.update_sharpen_immediate()

    def show_original(self):
        self.cancel_all_scheduled()
        self.hide_all_controls()
        self.preview_img = None
        self.hide_apply_button()
        self.show_image(self.original_img)

    def cancel_all_scheduled(self):
        if self._palette_after_id:
            self.root.after_cancel(self._palette_after_id)
            self._palette_after_id = None
        if self._median_after_id:
            self.root.after_cancel(self._median_after_id)
            self._median_after_id = None
        if self._laplacian_after_id:
            self.root.after_cancel(self._laplacian_after_id)
            self._laplacian_after_id = None
        if self._sharpen_after_id:
            self.root.after_cancel(self._sharpen_after_id)
            self._sharpen_after_id = None

    def show_sharpen_controls(self):
        if not self.sharpen_frame_visible:
            self.sharpen_strength_slider.pack(side=LEFT, padx=5)
            self.sharpen_frame_visible = True

    def hide_sharpen_controls(self):
        if self.sharpen_frame_visible:
            self.sharpen_strength_slider.pack_forget()
            self.sharpen_frame_visible = False

    def on_sharpen_strength_slider(self, _):
        self.schedule_sharpen_update()

    def schedule_sharpen_update(self):
        if self._sharpen_after_id:
            self.root.after_cancel(self._sharpen_after_id)
        self._sharpen_after_id = self.root.after(200, self.update_sharpen_immediate)

    def update_sharpen_immediate(self):
        if self.original_img is None:
            return
        k = "4" if self.laplacian_kernel_var.get() == 4 else "8"
        strength = int(self.sharpen_strength_slider.get())
        out = LaplacianSharpenFilter(kernel=k, strength=strength).apply(self.original_img)
        self.preview_img = out
        self.show_apply_button()
        self.show_image(self.preview_img)

    def show_apply_button(self):
        if not self.apply_visible:
            self.apply_button.pack(side=LEFT, padx=5)
            self.apply_visible = True

    def hide_apply_button(self):
        if self.apply_visible:
            self.apply_button.pack_forget()
            self.apply_visible = False

    def on_apply_button(self):
        if self.preview_img is None:
            return
        if self.original_img is not None:
            try:
                self.undo_stack.append(self.original_img.copy())
            except Exception:
                pass
        self.original_img = self.preview_img
        self.preview_img = None
        self.hide_apply_button()
        if self.median_frame_visible:
            try:
                min_val = int(self.median_slider.cget('from'))
            except Exception:
                min_val = 1
            self.median_slider.set(min_val)
            self._selection_active = True
        else:
            self._selection_active = False
        if self._doodle_controls_visible:
            self._doodle_active = True
        else:
            self._doodle_active = False
        self._draw = None
        self.clear_selection_overlay()
        self.roi = None
        self.show_image(self.original_img)

    def save_image(self):
        if self.original_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[
                                                ("PNG", "*.png"),
                                                ("JPEG", "*.jpg;*.jpeg"),
                                                ("BMP", "*.bmp"),
                                                ("TIFF", "*.tif;*.tiff")
                                            ])
        if not path:
            return
        self.original_img.save(path)

    def undo(self):
        if not self.undo_stack:
            return
        prev = self.undo_stack.pop()
        self.original_img = prev
        self.preview_img = None
        self.hide_apply_button()
        self._selection_active = False
        self._doodle_active = False
        self._draw = None
        self.clear_selection_overlay()
        self.roi = None
        self.show_image(self.original_img)

    def apply_scatter(self, cx, cy):
        if self.preview_img is None:
            return
        r = max(1, int(self.brush_size))
        w, h = self.preview_img.size
        n = max(3, int(self.scatter_density * 3.14159265 * r * r))
        for _ in range(n):
            t = random.random() * 6.2831853
            rr = r * (random.random() ** 0.5)
            x = int(round(cx + rr * math.cos(t)))
            y = int(round(cy + rr * math.sin(t)))
            if 0 <= x < w and 0 <= y < h:
                self.preview_img.putpixel((x, y), self.brush_color)

    def apply_stamp(self, cx, cy):
        if self.preview_img is None:
            return
        r = max(1, int(self.brush_size))
        w, h = self.preview_img.size
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(w, cx + r)
        y1 = min(h, cy + r)
        self._draw = self._draw or ImageDraw.Draw(self.preview_img)
        self._draw.ellipse([(x0, y0), (x1, y1)], fill=self.brush_color)

    def apply_path(self, x0, y0, x1, y1):
        r = max(1, int(self.brush_size))
        dx = x1 - x0
        dy = y1 - y0
        dist = max(1, int((dx*dx + dy*dy) ** 0.5))
        step = max(1, int(r // 3))
        steps = max(1, dist // step)
        for i in range(steps + 1):
            t = i / steps
            xi = int(round(x0 + t * dx))
            yi = int(round(y0 + t * dy))
            self.apply_stamp(xi, yi)

    def apply_scatter_path(self, x0, y0, x1, y1):
        r = max(1, int(self.brush_size))
        dx = x1 - x0
        dy = y1 - y0
        dist = max(1, int((dx*dx + dy*dy) ** 0.5))
        step = max(1, int(r // 3))
        steps = max(1, dist // step)
        for i in range(steps + 1):
            t = i / steps
            xi = int(round(x0 + t * dx))
            yi = int(round(y0 + t * dy))
            self.apply_scatter(xi, yi)

    # duplicate undo method removed

if __name__ == "__main__":
    root = Tk()
    app = ImageFilterApp(root)
    root.mainloop()