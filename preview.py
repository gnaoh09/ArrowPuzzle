import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patches as mpatches

plt.rcParams['toolbar'] = 'toolbar2'

def convert_y(game_y, H):
    """Giữ nguyên hệ toạ độ game"""
    return game_y

def get_arrow_direction(nodes):
    """Lấy hướng từ 2 node đầu tiên"""
    if len(nodes) < 2:
        return "UNKNOWN"
    
    x1, y1 = nodes[0]["x"], nodes[0]["y"]
    x2, y2 = nodes[1]["x"], nodes[1]["y"]
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 1:
        return "UP"
    elif dx == 0 and dy == -1:
        return "DOWN"
    elif dx == -1 and dy == 0:
        return "LEFT"
    elif dx == 1 and dy == 0:
        return "RIGHT"
    
    return "UNKNOWN"

def get_color_palette(num_colors):
    """Tự động tạo bảng màu cho số lượng mũi tên bất kỳ"""
    import colorsys
    
    # Nếu ít hơn 16 màu, dùng bảng màu định sẵn
    base_colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#FFD93D',  # Yellow
        '#95E1D3',  # Mint
        '#A8E6CF',  # Light green
        '#FFB6C1',  # Pink
        '#DDA15E',  # Brown
        '#B4A7D6',  # Purple
        '#FF9F45',  # Orange
        '#6C5CE7',  # Indigo
        '#00B894',  # Green
        '#E17055',  # Coral  
        '#0984E3',  # Blue
        '#FDCB6E',  # Light Yellow
        '#C4E538',  # Lime
        '#E84393',  # Magenta
    ]
    
    if num_colors <= len(base_colors):
        return {i: base_colors[i] for i in range(num_colors)}
    
    # Nếu cần nhiều màu hơn, tự động sinh thêm bằng HSV
    colors = {}
    for i in range(num_colors):
        if i < len(base_colors):
            colors[i] = base_colors[i]
        else:
            # Sinh màu mới bằng cách phân bổ đều trên vòng tròn màu
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio cho phân bố đều
            saturation = 0.6 + (i % 3) * 0.15  # Thay đổi độ bão hòa
            value = 0.85 + (i % 2) * 0.1  # Thay đổi độ sáng
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[i] = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
    
    return colors

def draw_level_preview(level_data, save_path=None, show=True, interactive=True):
    """Vẽ preview level với các mũi tên và đường đi"""
    
    W = level_data["size"]["x"]
    H = level_data["size"]["y"]
    
    # Tạo figure với kích thước phù hợp (tối đa 16 inch)
    fig_width = min(W * 0.8, 16)
    fig_height = min(H * 0.8, 12)
    
    use_tkinter = False
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect('equal')
    
    # Vẽ grid
    for x in range(W):
        ax.axvline(x - 0.5, color='#E0E0E0', linewidth=1)
    for y in range(H):
        ax.axhline(y - 0.5, color='#E0E0E0', linewidth=1)
    
    # Vẽ background cho các ô
    for x in range(W):
        for y in range(H):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                facecolor='#FAFAFA', 
                                edgecolor='#CCCCCC', 
                                linewidth=0.5)
            ax.add_patch(rect)
    
    # Lấy bảng màu
    num_arrows = len(level_data["arrows"])
    color_palette = get_color_palette(num_arrows)
    
    # Vẽ các arrow
    for idx, arrow in enumerate(level_data["arrows"]):
        nodes = arrow["nodes"]
        color_idx = arrow.get("color", idx)
        color = color_palette.get(color_idx, '#999999')
        
        # Chuyển đổi toạ độ
        xs = [n["x"] for n in nodes]
        ys = [convert_y(n["y"], H) for n in nodes]
        
        # Vẽ đường đi (path) với độ dày lớn hơn
        ax.plot(xs, ys, color=color, linewidth=4, alpha=0.7, zorder=2)
        
        # Vẽ các điểm trên đường đi
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i == 0:
                # Điểm bắt đầu (vẽ mũi tên)
                circle = Circle((x, y), 0.15, color=color, zorder=3)
                ax.add_patch(circle)
            else:
                # Các điểm trung gian
                circle = Circle((x, y), 0.08, color=color, alpha=0.6, zorder=3)
                ax.add_patch(circle)
        
        # Vẽ đầu mũi tên từ node cuối-1 đến node cuối
        if len(nodes) >= 2:
            x1 = nodes[-2]["x"]
            y1 = convert_y(nodes[-2]["y"], H)
            x2 = nodes[-1]["x"]
            y2 = convert_y(nodes[-1]["y"], H)
            
            # Vẽ mũi tên
            arrow_patch = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', 
                mutation_scale=25,
                color=color,
                linewidth=4,
                zorder=4
            )
            ax.add_patch(arrow_patch)
        
        # Hiển thị hướng và thông tin tại điểm bắt đầu
        if nodes:
            start_x = nodes[0]["x"]
            start_y = convert_y(nodes[0]["y"], H)
            direction = get_arrow_direction(nodes)
            
            # Vẽ text box với hướng
            direction_symbol = {
                "UP": "↑",
                "DOWN": "↓", 
                "LEFT": "←",
                "RIGHT": "→",
                "UNKNOWN": "?"
            }
            
            symbol = direction_symbol.get(direction, "?")
            
            # Vẽ background cho text
            bbox_props = dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor=color,
                            linewidth=2,
                            alpha=0.9)
            
            ax.text(start_x, start_y, f"#{idx}", 
                   fontsize=10, 
                   fontweight='bold',
                   ha='center', 
                   va='center',
                   color=color,
                   bbox=bbox_props,
                   zorder=5)
    
    # Vẽ toạ độ ở các góc của grid
    for x in range(W):
        for y in range(H):
            ax.text(x - 0.45, y - 0.45, f"({x},{y})", 
                   fontsize=6, 
                   color='#999999',
                   ha='left',
                   va='bottom',
                   zorder=1)
    
    # Tiêu đề và thông tin
    title = level_data.get("name", "Level Preview")
    time_limit = level_data.get("timeLimit", "N/A")
    num_arrows = len(level_data.get("arrows", []))
    
    ax.set_title(f"{title}\nSize: {W}x{H} | Time: {time_limit}s | Arrows: {num_arrows}", 
                fontsize=14, 
                fontweight='bold',
                pad=20)
    
    # Thêm legend
    legend_elements = []
    for idx, arrow in enumerate(level_data["arrows"]):
        color_idx = arrow.get("color", idx)
        color = color_palette.get(color_idx, '#999999')
        direction = get_arrow_direction(arrow["nodes"])
        length = len(arrow["nodes"])
        
        label = f"Arrow #{idx}: {direction} (len={length})"
        legend_elements.append(mpatches.Patch(color=color, label=label))
    
    # Chỉ hiển thị legend nếu không quá nhiều arrows
    if num_arrows <= 20:
        ax.legend(handles=legend_elements, 
                 loc='center left', 
                 bbox_to_anchor=(1, 0.5),
                 fontsize=8)
    
    # Tắt tick marks
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        # plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()

# ----------------------------------------
# Main execution
# ----------------------------------------
if __name__ == "__main__":
    with open("/Users/hoangnguyen/Documents/py/ArrowPuzzle/output_full_3/lv80_original.json", "r") as f:
        level = json.load(f)
    draw_level_preview(level, save_path="level_preview.png", show=True, interactive=False)
   