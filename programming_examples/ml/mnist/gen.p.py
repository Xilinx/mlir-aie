from ipycanvas import Canvas
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import os
from PIL import Image

def create_data_collection_canvas():
    # Create canvas with sync_image_data enabled
    canvas = Canvas(width=280, height=280, sync_image_data=True)
    
    # Set white background
    canvas.fill_style = 'white'
    canvas.fill_rect(0, 0, 280, 280)
    
    # Drawing state
    drawing = False
    
    # Data collection state
    current_digit = 0
    digit_count = 0
    max_per_digit = 5
    
    def on_mouse_down(x, y):
        nonlocal drawing
        drawing = True
        canvas.fill_style = 'black'
        canvas.fill_circle(x, y, 4)
    
    def on_mouse_move(x, y):
        if drawing:
            canvas.fill_style = 'black'
            canvas.fill_circle(x, y, 4)
    
    def on_mouse_up(x, y):
        nonlocal drawing
        drawing = False
    
    def clear_canvas():
        canvas.fill_style = 'white'
        canvas.fill_rect(0, 0, 280, 280)
        # No border - just clean white canvas
    
    def save_digit():
        nonlocal current_digit, digit_count
        
        # Save the canvas to a file
        canvas.to_file("temp_canvas.png")
        
        # Load the saved image using PIL
        img = Image.open("temp_canvas.png").convert('L')  # Convert to grayscale
        
        # Resize to 28x28
        img_resized = img.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(img_resized) / 255.0
        
        # INVERT the image (black drawing on white background -> white drawing on black background)
        img_array = 1.0 - img_array
        
        # Normalize (MNIST style)
        mean = 0.1307
        std = 0.3081
        img_normalized = (img_array - mean) / std
        
        # Flatten and crop to 768 features
        img_flat = img_normalized.flatten()
        img_cropped = img_flat[:768]
        
        # Create custom data directory
        os.makedirs("custom_data", exist_ok=True)
        
        # Save the processed image
        filename = f"custom_data/digit_{current_digit}_{digit_count:02d}.npy"
        np.save(filename, img_cropped)
        
        # Also save a visual version
        visual_filename = f"custom_data/digit_{current_digit}_{digit_count:02d}_visual.png"
        img_vis = np.clip((img_normalized + mean) / std, 0, 1)
        Image.fromarray((img_vis * 255).astype(np.uint8)).save(visual_filename)
        
        print(f"✅ Saved digit {current_digit} (#{digit_count + 1}) as {filename}")
        
        # Update counters
        digit_count += 1
        if digit_count >= max_per_digit:
            digit_count = 0
            current_digit += 1
            if current_digit > 9:
                current_digit = 0
                print("🎉 All digits completed! Starting over...")
        
        # Clear canvas for next digit
        clear_canvas()
        
        # Update status display
        update_status()
    
    def update_status():
        # Update the status label (not drawn on canvas)
        status_label.value = f"Current: Digit {current_digit} | Count: {digit_count + 1}/{max_per_digit}"
    
    # Connect mouse events
    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_up(on_mouse_up)
    
    # Create buttons
    clear_button = widgets.Button(description='Clear', button_style='warning')
    save_button = widgets.Button(description='Save Digit', button_style='success')
    next_button = widgets.Button(description='Next Digit', button_style='info')
    
    # Create status label
    status_label = widgets.Label(value="Current: Digit 0 | Count: 1/5")
    
    def on_clear(b):
        clear_canvas()
        update_status()
    
    def on_save(b):
        save_digit()
    
    def on_next(b):
        nonlocal current_digit, digit_count
        current_digit += 1
        if current_digit > 9:
            current_digit = 0
        digit_count = 0
        clear_canvas()
        update_status()
        print(f"➡️ Moved to digit {current_digit}")
    
    # Connect buttons
    clear_button.on_click(on_clear)
    save_button.on_click(on_save)
    next_button.on_click(on_next)
    
    # Initial status
    update_status()
    
    display(widgets.VBox([
        widgets.HTML("<h3>Data Collection: Draw 5 examples of each digit (0-9)</h3>"),
        status_label,  # Status outside the canvas
        canvas,
        widgets.HBox([clear_button, save_button, next_button])
    ], layout=widgets.Layout(
        justify_content='center',
        align_items='center',
        width='100%'
    )))
    
    return canvas

# Create the data collection canvas
data_collection_canvas = create_data_collection_canvas()