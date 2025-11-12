import cv2

def find_working_cameras():
    """Tests camera indices 0 through 4."""
    print("Searching for available cameras...")
    
    for i in range(5):
        # --- Test with default backend ---
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[SUCCESS] Index {i} (Default) is working!")
            cap.release()
        else:
            print(f"[INFO] Index {i} (Default) is NOT available.")

        # --- Test with DSHOW backend ---
        cap_dshow = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if cap_dshow.isOpened():
            print(f"[SUCCESS] Index {i} (DSHOW) is working!")
            cap_dshow.release()
        else:
            print(f"[INFO] Index {i} (DSHOW) is NOT available.")
    
    print("Search complete.")

if __name__ == "__main__":
    find_working_cameras()