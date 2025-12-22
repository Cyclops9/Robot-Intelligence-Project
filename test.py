import os
if __name__ == "__main__":
    # Test it with your baseline video
    test_video = "rl-video-step-0.mp4"
    if os.path.exists(test_video):
        print(f"--> Analyzing: {test_video}")
    else:
        print(f"Video not found at {test_video}. Run a baseline training first!")