import cv2

def create_a4_charuco():
    # 300 DPI precise pixel sizing for A4 Paper (210mm x 297mm)
    # 210mm / 25.4 * 300 = ~2480 pixels width
    # 297mm / 25.4 * 300 = ~3508 pixels height
    a4_width_px = 2480
    a4_height_px = 3508
    
    # Grid properties (9 squares along the long edge, 7 along the short edge)
    # Square length = 0.025 meters (25mm)
    # Marker length = 0.019 meters (19mm)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((7, 9), 0.025, 0.019, dictionary)
    
    print("Generating crisp, high-res A4 calibration image...")
    board_img = board.generateImage((a4_width_px, a4_height_px), marginSize=120)
    
    output_filename = "charuco_A4_25mm.png"
    cv2.imwrite(output_filename, board_img)
    print(f"Success! Saved as '{output_filename}'.")
    print("⚠️  CRITICAL PRINT SETTING: When printing this PNG, select 'Actual Size' (100% Scale). Do NOT select 'Fit to Page'.")

if __name__ == "__main__":
    create_a4_charuco()