import os
import sys
import traceback

def debug_info():
    print("=== ДИАГНОСТИКА ===")
    print(f"Python: {sys.version}")
    print(f"Current dir: {os.getcwd()}")
    print(f"EXE location: {sys.executable if hasattr(sys, 'frozen') else __file__}")
    
    print("\n=== ПРОВЕРКА ФАЙЛОВ ===")
    # Проверяем наличие файлов
    files_to_check = [
        'templates/index.html',
        'models/best_lung_model.h5', 
        'static/uploads',
        'diagnosis_history.json'
    ]
    
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"{file}: {'✅ СУЩЕСТВУЕТ' if exists else '❌ ОТСУТСТВУЕТ'}")
        if exists:
            print(f"   Размер: {os.path.getsize(file) if os.path.isfile(file) else 'папка'}")

    print("\n=== ПРОВЕРКА БИБЛИОТЕК ===")
    # Пробуем импортировать библиотеки
    libs = ['flask', 'tensorflow', 'PIL', 'numpy', 'sklearn']
    
    for lib in libs:
        try:
            __import__(lib)
            print(f"{lib}: ✅")
        except Exception as e:
            print(f"{lib}: ❌ - {e}")

    print("\n=== СОДЕРЖИМОЕ ПАПКИ ===")
    for item in os.listdir('.'):
        print(f"  {item}")

    input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    debug_info()