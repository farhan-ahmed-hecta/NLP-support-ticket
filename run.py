"""
=============================================================================
  AI Support Ticket Classification & Triage System
  Main Execution Entry Point for Visual Studio / VS Code & Localhost
=============================================================================
"""
import sys
import os
import time
import webbrowser

# Ensure current working directory is set to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

def check_dependencies():
    """Verify that all core libraries are installed."""
    required = ["sklearn", "joblib", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n" + "!" * 65)
        print(" [WARNING] Some required dependencies are missing:")
        for m in missing:
            print(f"   - {m}")
        print("\n Please install them by running:")
        print("   pip install -r requirements.txt")
        print("!" * 65 + "\n")
        return False
    return True

def main():
    print("=" * 65)
    print("   📧 AI SUPPORT TICKET CLASSIFICATION & TRIAGE SYSTEM")
    print("=" * 65)
    print(f"[INFO] Working Directory: {PROJECT_ROOT}")
    
    if not check_dependencies():
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    print("[INFO] Launching local web server...")
    try:
        import server
        server.run_server(port=8080, open_browser=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to run server: {e}")
        input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()
