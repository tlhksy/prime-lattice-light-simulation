"""
PNG Fixer for LaTeX
-------------------
Bu script PNG dosyalarını LaTeX ile uyumlu hale getirir.

Kullanım:
1. Bu dosyayı figürlerinizin olduğu klasöre koyun
2. Çalıştırın: python fix_png_for_latex.py
"""

import os
import sys

# Pillow kurulu değilse kur
try:
    from PIL import Image
except ImportError:
    print("Pillow kuruluyor...")
    os.system(f"{sys.executable} -m pip install Pillow")
    from PIL import Image

def fix_png_files(folder=None):
    """PNG dosyalarını düzelt ve LaTeX uyumlu yap"""
    
    if folder is None:
        folder = os.getcwd()
    
    # Düzeltilecek dosyalar
    target_files = [
        'fig1_patterns',
        'fig2_metrics', 
        'fig3_profiles',
        'fig4_fft',
        'validation_report'
    ]
    
    print(f"\n{'='*60}")
    print("PNG FIXER FOR LATEX")
    print(f"{'='*60}")
    print(f"Klasör: {folder}\n")
    
    fixed_count = 0
    
    for base_name in target_files:
        # Farklı uzantı kombinasyonlarını dene
        possible_names = [
            base_name,
            base_name + '.png',
            base_name + '.PNG',
            base_name + '.Png',
        ]
        
        found = False
        for name in possible_names:
            full_path = os.path.join(folder, name)
            
            if os.path.exists(full_path):
                found = True
                try:
                    # Dosyayı aç
                    img = Image.open(full_path)
                    original_format = img.format
                    original_mode = img.mode
                    
                    # RGBA ise RGB'ye çevir (bazı LaTeX sürümleri için)
                    if img.mode == 'RGBA':
                        # Beyaz arka plan oluştur
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Yeni dosya adı
                    output_path = os.path.join(folder, base_name + '.png')
                    
                    # PNG olarak kaydet (optimize edilmiş)
                    img.save(output_path, 'PNG', optimize=True)
                    
                    print(f"✓ {base_name}.png")
                    print(f"  Orijinal: {original_format}, {original_mode}")
                    print(f"  Yeni: PNG, RGB")
                    
                    fixed_count += 1
                    
                except Exception as e:
                    print(f"✗ {base_name}: HATA - {e}")
                
                break
        
        if not found:
            print(f"? {base_name}: Dosya bulunamadı")
    
    print(f"\n{'='*60}")
    print(f"Toplam düzeltilen: {fixed_count}/{len(target_files)}")
    print(f"{'='*60}")
    
    if fixed_count == len(target_files):
        print("\n✓ Tüm dosyalar düzeltildi!")
        print("Şimdi LaTeX'i tekrar derleyebilirsiniz:")
        print("  pdflatex paper_v3_validated.tex")
    else:
        print("\n⚠ Bazı dosyalar eksik veya hatalı.")
        print("Simülasyonu tekrar çalıştırın:")
        print("  python light_simulation_v4.5_final.py")

def check_png_validity(folder=None):
    """PNG dosyalarının geçerliliğini kontrol et"""
    
    if folder is None:
        folder = os.getcwd()
    
    print(f"\n{'='*60}")
    print("PNG VALİDASYON KONTROLÜ")
    print(f"{'='*60}\n")
    
    for f in os.listdir(folder):
        if f.lower().endswith('.png'):
            full_path = os.path.join(folder, f)
            try:
                img = Image.open(full_path)
                img.verify()  # Dosya bütünlüğünü kontrol et
                
                # Tekrar aç (verify sonrası gerekli)
                img = Image.open(full_path)
                print(f"✓ {f}: {img.format} {img.size} {img.mode}")
                
            except Exception as e:
                print(f"✗ {f}: GEÇERSİZ - {e}")

if __name__ == "__main__":
    # Mevcut klasörde çalıştır
    folder = os.getcwd()
    
    # Argüman verilmişse onu kullan
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    
    print("\n1. PNG dosyalarını kontrol ediliyor...")
    check_png_validity(folder)
    
    print("\n2. PNG dosyaları düzeltiliyor...")
    fix_png_files(folder)
