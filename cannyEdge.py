import sys
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def main(filepath, filename):
    assert os.path.exists(filepath), f"{filepath} does not exist"
    
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    assert img is not None, "File could not be read"
    
    edges = cv.Canny(img, 100, 200)
    edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    cv.imwrite(filename, edges)
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py [filepath] [filename]")
        sys.exit(1)
        
    filepath = sys.argv[1]
    filename = sys.argv[2] + ".png"
    
    main(filepath, filename)