import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

def analyze_arrow_structure(json_path):
    """
    Phân tích cấu trúc thực tế của mũi tên trong game
    """
    
    with open(json_path, "r") as f:
        level = json.load(f)
    
    print("=" * 70)
    print("PHÂN TÍCH CẤU TRÚC MŨI TÊN TRONG GAME")
    print("=" * 70)
    print()
    
    # Phân tích một vài arrows đầu tiên
    for idx in range(min(5, len(level["arrows"]))):
        arrow = level["arrows"][idx]
        nodes = arrow["nodes"]
        
        print(f"Arrow #{idx}:")
        print(f"  Color: {arrow.get('color', 'N/A')}")
        print(f"  Số nodes: {len(nodes)}")
        print(f"  Đường đi:")
        
        for i, node in enumerate(nodes):
            marker = ""
            if i == 0:
                marker = " <- ĐIỂM ĐẦU (first)"
            elif i == len(nodes) - 1:
                marker = " <- ĐIỂM CUỐI (last)"
            print(f"    [{i}] ({node['x']}, {node['y']}){marker}")
        
        # Tính hướng theo 2 cách
        if len(nodes) >= 2:
            # Cách 1: Từ đầu tiên đến thứ 2
            x1, y1 = nodes[0]["x"], nodes[0]["y"]
            x2, y2 = nodes[1]["x"], nodes[1]["y"]
            dx1, dy1 = x2 - x1, y2 - y1
            
            # Cách 2: Từ cuối-1 đến cuối
            x3, y3 = nodes[-2]["x"], nodes[-2]["y"]
            x4, y4 = nodes[-1]["x"], nodes[-1]["y"]
            dx2, dy2 = x4 - x3, y4 - y3
            
            dir1 = get_direction_name(dx1, dy1)
            dir2 = get_direction_name(dx2, dy2)
            
            print(f"  Hướng từ [0]->[1]:      {dir1}")
            print(f"  Hướng từ [-2]->[-1]:    {dir2}")
            
            if dir1 == dir2:
                print(f"  => Cả 2 cách: {dir1}")
            else:
                print(f"  ⚠️  KHÁC NHAU! Cần xác định đúng!")
        
        print()
    
    print("=" * 70)
    print("KẾT LUẬN:")
    print("=" * 70)
    print()
    print("Trong game Arrow Puzzle:")
    print("1. MŨI TÊN (arrow head) nằm ở VỊ TRÍ NÀO?")
    print("   - Nếu ở nodes[0] (đầu tiên) => Hướng = [0]->[1]")
    print("   - Nếu ở nodes[-1] (cuối cùng) => Hướng = [-2]->[-1]")
    print()
    print("2. Dựa vào logic game:")
    print("   - Arrow BLOCKS các arrow khác trên đường đi của nó")
    print("   - Đường đi BẮT ĐẦU từ đâu? KẾT THÚC ở đâu?")
    print()
    print("Hãy visualize để xác định chính xác!")
    print()

def get_direction_name(dx, dy):
    """Chuyển dx, dy thành tên hướng"""
    if dx > 0:
        return "RIGHT"
    elif dx < 0:
        return "LEFT"
    elif dy > 0:
        return "UP"
    elif dy < 0:
        return "DOWN"
    return "UNKNOWN"

def visualize_arrow_detail(json_path, arrow_index=0):
    """
    Vẽ chi tiết 1 arrow để thấy rõ hướng
    """
    with open(json_path, "r") as f:
        level = json.load(f)
    
    arrow = level["arrows"][arrow_index]
    nodes = arrow["nodes"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tìm bounds
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    
    for ax, title, arrow_at in [(ax1, "Mũi tên ở ĐIỂM ĐẦU [0]", "start"), 
                                 (ax2, "Mũi tên ở ĐIỂM CUỐI [-1]", "end")]:
        
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Vẽ đường đi
        ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.5, label='Path')
        
        # Vẽ các điểm
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i == 0:
                color = 'green'
                size = 200
                label = 'Start [0]'
            elif i == len(nodes) - 1:
                color = 'red'
                size = 200
                label = 'End [-1]'
            else:
                color = 'blue'
                size = 100
                label = None
            
            ax.scatter(x, y, c=color, s=size, zorder=5, edgecolor='black', linewidth=2, label=label)
            ax.text(x + 0.15, y + 0.15, f"[{i}]", fontsize=10, fontweight='bold')
        
        # Vẽ mũi tên
        if arrow_at == "start" and len(nodes) >= 2:
            # Mũi tên từ [0] -> [1]
            x1, y1 = nodes[0]["x"], nodes[0]["y"]
            x2, y2 = nodes[1]["x"], nodes[1]["y"]
            arrow_patch = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', 
                mutation_scale=30,
                color='darkgreen',
                linewidth=4,
                zorder=10,
                label='Arrow Head'
            )
            ax.add_patch(arrow_patch)
            
            direction = get_direction_name(x2 - x1, y2 - y1)
            ax.text(x1, y1 - 0.5, f"Direction: {direction}", 
                   fontsize=12, fontweight='bold', 
                   ha='center', color='darkgreen',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
        elif arrow_at == "end" and len(nodes) >= 2:
            # Mũi tên từ [-2] -> [-1]
            x1, y1 = nodes[-2]["x"], nodes[-2]["y"]
            x2, y2 = nodes[-1]["x"], nodes[-1]["y"]
            arrow_patch = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', 
                mutation_scale=30,
                color='darkred',
                linewidth=4,
                zorder=10,
                label='Arrow Head'
            )
            ax.add_patch(arrow_patch)
            
            direction = get_direction_name(x2 - x1, y2 - y1)
            ax.text(x2, y2 - 0.5, f"Direction: {direction}", 
                   fontsize=12, fontweight='bold', 
                   ha='center', color='darkred',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
    
    plt.suptitle(f"Arrow #{arrow_index} - So sánh 2 cách hiểu hướng mũi tên", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('arrow_direction_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Đã lưu visualization: arrow_direction_analysis.png")
    plt.show()

def check_all_arrows_consistency(json_path):
    """
    Kiểm tra xem tất cả arrows có consistent về hướng không
    (đường đi luôn đi theo 1 hướng từ đầu đến cuối)
    """
    with open(json_path, "r") as f:
        level = json.load(f)
    
    print("\n" + "=" * 70)
    print("KIỂM TRA TÍNH NHẤT QUÁN CỦA HƯỚNG ĐI")
    print("=" * 70 + "\n")
    
    inconsistent = []
    
    for idx, arrow in enumerate(level["arrows"]):
        nodes = arrow["nodes"]
        if len(nodes) < 2:
            continue
        
        # Lấy hướng từ đoạn đầu và đoạn cuối
        dir_first = get_direction_name(
            nodes[1]["x"] - nodes[0]["x"],
            nodes[1]["y"] - nodes[0]["y"]
        )
        dir_last = get_direction_name(
            nodes[-1]["x"] - nodes[-2]["x"],
            nodes[-1]["y"] - nodes[-2]["y"]
        )
        
        if dir_first != dir_last:
            inconsistent.append({
                "index": idx,
                "first_dir": dir_first,
                "last_dir": dir_last,
                "nodes": nodes
            })
    
    if not inconsistent:
        print("✅ TẤT CẢ arrows đều đi theo 1 hướng nhất quán!")
        print("   => Có thể dùng [0]->[1] HOẶC [-2]->[-1] đều được")
    else:
        print(f"⚠️  Có {len(inconsistent)} arrows KHÔNG nhất quán:")
        for item in inconsistent[:5]:
            print(f"   Arrow #{item['index']}: Đầu={item['first_dir']}, Cuối={item['last_dir']}")
        print("\n   => Arrows có thể rẽ/quẹo! Cần xem logic game cụ thể.")
    
    return len(inconsistent) == 0

if __name__ == "__main__":
    json_path = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/asset-game-level/lv8.json"
    
    # Bước 1: Phân tích cấu trúc
    analyze_arrow_structure(json_path)
    
    # Bước 2: Kiểm tra tính nhất quán
    is_consistent = check_all_arrows_consistency(json_path)
    
    # Bước 3: Visualize chi tiết 1 arrow
    print("\n" + "=" * 70)
    print("VISUALIZE CHI TIẾT")
    print("=" * 70)
    print("\nĐang vẽ Arrow #0 để so sánh 2 cách hiểu...")
    visualize_arrow_detail(json_path, arrow_index=0)
    
    print("\n" + "=" * 70)
    print("KHUYẾN NGHỊ")
    print("=" * 70)
    print()
    print("Để xác định CHÍNH XÁC, bạn cần kiểm tra:")
    print("1. Trong game, khi di chuyển arrow, nó di chuyển THEO HƯỚNG NÀO?")
    print("2. Arrow HEAD (đầu mũi tên) thực sự ở đâu trong game UI?")
    print("3. Logic blocking: Arrow bắn/di chuyển từ đâu đến đâu?")
    print()
    print("Dựa vào đó, chọn:")
    print("  - Nếu arrow HEAD ở nodes[0]: Hướng = [0] -> [1]")
    print("  - Nếu arrow HEAD ở nodes[-1]: Hướng = [-2] -> [-1]")
    print("=" * 70)