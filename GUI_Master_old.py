import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNNModel
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as PdfImage, SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
import datetime
from PIL import Image as PILImage
import pyttsx3 
import threading


from keras.models import load_model
import numpy as np
from PIL import Image

global fn
fn = ""

def speak(text):
    engine.say(text)
    engine.runAndWait()
def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()
engine = pyttsx3.init()
engine.say("Login successful")
engine.runAndWait()


root = tk.Tk()
root.configure(background="seashell2")
# root.geometry("1300x700")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Lung cancer disease Detection using ML ")

# For background Image
image2 = Image.open('l8.jpg')
image2 = image2.resize((w, h), Image.LANCZOS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)

lbl = tk.Label(root, text="Welcome to Detection System of Lung Cancer", font=('times', 25, ' bold '), height=1, width=80,
               bg="white", fg="black")
lbl.place(relx=0.5, y=10, anchor='n')


frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '), bg="black", fg="white")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=120)


def train_model():
    update_label("Model Training Start...............")
    start = time.time()
    X = CNNModel.main()
    end = time.time()
    ET = "Execution Time: {0:.4} seconds \n".format(end - start)
    msg = "Model Training Completed.." + '\n' + X + '\n' + ET
    print(msg)


def test_model_proc(fn):
    global detection_result
    IMAGE_SIZE = 64  # Image size used during training
    
    print(f"Testing with image: {fn}")
    
    if fn:
        try:
            # Load the pre-trained model
            model = load_model('lung_model.h5')
            print("Model loaded successfully.")

            # Open the image file
            img = Image.open(fn)

            # Resize the image to the required size
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            
            # Convert the image to a numpy array and normalize it
            img = np.array(img)
            img = img.astype('float32')  # Ensure the image is float32 for model prediction
            img = img / 255.0  # Normalize the image to the range [0, 1]
            
            # Check if the image is grayscale and convert to 3 channels (RGB)
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB by stacking the same channel
            elif img.shape[-1] != 3:
                print("Error: Image does not have 3 channels. Exiting.")
                return
            
            # Reshape the image to fit the model input
            img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

            # Print the image shape for debugging
            print("Image shape after preprocessing:", img.shape)
            
            # Make the prediction
            prediction = model.predict(img)

            # Print the prediction probabilities
            print("Prediction probabilities:", prediction)

            # Get the class with the highest probability
            plant = np.argmax(prediction)
            print(f"Predicted class index: {plant}")

            # Check for additional size-specific categories, e.g., "Small" or "Large"
            if plant == 0:
                detection_result = "Benign Lung Cancer Detected (Small)"
            elif plant == 1:
                detection_result = "Malignant Lung Cancer Detected (Large)"
            elif plant == 2:
                detection_result = "Normal Lung Cancer Detected"
            else:
                detection_result = "Unknown class detected"
            
            print("Detection result:", detection_result)
            return detection_result
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return "Error during prediction"

    else:
        print("No image file provided.")
        return "No image file provided."

        


def update_label(str_T):
    result_label = tk.Label(root, text=str_T, width=50, font=("bold", 20), bg='#E5E4E2', fg='black')
    result_label.place(x=300, y=450)
    


def test_model():
    global fn
    if fn != "":
        
        update_label("Model Testing Start...............")
        start = time.time()
        X = test_model_proc(fn)
        x2 = format(X) + " Disease is detected"
        end = time.time()
        

        ET = "Execution Time: {0:.4} seconds \n".format(end - start)
        msg = "Image Testing Completed.." + '\n' + x2 + '\n' + ET
        fn = ""
    else:
        speak_async("select image for prediction")
        msg = "Please Select Image For Prediction...."
        
    speak_async(msg)
    update_label(msg)


def openimage():
    global stored_image  # Global variable to store the original image
    global fn
    
    # Open file dialog to select the image
    fileName = askopenfilename(initialdir='D:/23 Protech/100% code/Lung Cancer/lung cancer 100%/test_set', 
                               title='Select image for Analysis ', 
                               filetypes=[("all files", "*.*")])
    
    IMAGE_SIZE = 200  # Resize the image to 200x200
    imgpath = fileName
    fn = fileName  # Store the file name for later use
    
    # Open the image
    img = Image.open(imgpath)
    img = img.resize((IMAGE_SIZE, 200))  # Resize image
    
    # Store the PIL Image object for later use
    stored_image = img  # This will store the original PIL Image
    
    # Convert to numpy array for further processing
    img = np.array(img)
    
    # Image dimensions
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    # Prepare the image for display
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)

    # Display the image on the GUI
    img_label = tk.Label(root, image=imgtk, height=250, width=250)
    img_label.image = imgtk
    img_label.place(x=300, y=100)  # Adjust position as needed


def convert_grey():
    global fn
    IMAGE_SIZE = 200
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE, 200))
    img = np.array(img)
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)
    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    img2 = tk.Label(root, image=imgtk, height=250, width=250, bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)
    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    engine.say("Preprocessing Done")
    engine.runAndWait()

def generate_report():
    global detection_result, stored_image

    try:
        with open("logged_in_user.txt", "r") as f:
            logged_in_user = f.read().strip()
    except FileNotFoundError:
        logged_in_user = "Unknown"

    if detection_result == "":
        print("No detection result found. Please run detection first.")
        return

    # Educational advice based on result
    if "Benign" in detection_result:
        advice_title = "Diagnosis: Benign Lung Tumor (Small)"
        advice = (
            "<b>Precautions:</b><br/>"
            "• Schedule regular follow-ups and imaging.<br/>"
            "• Avoid smoking, vaping, and polluted environments.<br/>"
            "• Maintain a healthy diet rich in antioxidants.<br/>"
            "• Practice breathing exercises and light physical activity.<br/><br/>"
            "<b>Curability:</b><br/>"
            "Benign tumors can often be removed via minor surgery or monitored over time. "
            "Treatment is usually not urgent unless symptoms occur or the tumor grows."
        )
    elif "Malignant" in detection_result:
        advice_title = "Diagnosis: Malignant Lung Tumor (Large)"
        advice = (
            "<b>Precautions:</b><br/>"
            "• Seek medical attention immediately.<br/>"
            "• Avoid all tobacco and second-hand smoke.<br/>"
            "• Strengthen your immune system with a balanced diet and rest.<br/>"
            "• Avoid air pollutants and occupational hazards.<br/><br/>"
            "<b>Curability:</b><br/>"
            "Depends on the stage and location. Early-stage tumors are often curable. "
            "Advanced stages require long-term treatment and management."
        )
    elif "Normal" in detection_result:
        advice_title = "Diagnosis: Normal Lung Condition"
        advice = (
            "<b>Precautions:</b><br/>"
            "• Continue avoiding smoking and pollution.<br/>"
            "• Stay active and maintain a healthy lifestyle.<br/>"
            "• Regular checkups recommended if there’s a family history.<br/><br/>"
            "<b>Curability:</b><br/>"
            "No signs of disease detected. Stay consistent with healthy habits."
        )
    else:
        advice_title = "Diagnosis: Unknown"
        advice = "No specific advice available."

    filename = "lung_cancer_report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)

    # Add page border
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.rect(20, 20, letter[0] - 40, letter[1] - 40)

    # Clinic Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, 750, "LungScan Diagnostics Center")
    c.setFont("Helvetica", 12)
    c.drawString(180, 735, "Comprehensive Lung Imaging Report")

    # Line separator
    c.line(50, 720, 550, 720)

    # Patient Info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 700, "Patient Name:")
    c.drawString(250, 700, logged_in_user)

    c.drawString(50, 680, "Date of Report:")
    c.drawString(250, 680, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    c.drawString(50, 660, "Scan Result:")
    c.setFont("Helvetica", 12)
    c.drawString(250, 660, detection_result)

    # Add image (resize accordingly)
    temp_img_path = "temp_lung_image.jpg"
    stored_image.save(temp_img_path)
    c.drawImage(temp_img_path, 50, 420, width=200, height=200)

    # Add section for Advice/Diagnosis
    from reportlab.platypus import Frame, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4, fontSize=10))

    frame = Frame(270, 100, 280, 500, showBoundary=0)
    story = [
        Spacer(1, 2),
        Paragraph("<b>Summary of Findings:</b>", styles['Heading4']),
        Paragraph("The scan analysis indicates a preliminary classification of the uploaded lung X-ray. "
                  "The AI model has detected patterns consistent with the above diagnosis based on training from thousands of radiographic images.", styles['Justify']),
        Spacer(1, 10),
        Paragraph("<b>What Does This Mean?</b>", styles['Heading4']),
        Paragraph("This report provides a first-level analysis and does not replace expert radiological or clinical evaluation. "
                  "However, it can serve as a quick preliminary check for early action.", styles['Justify']),
        Spacer(1, 10),
        Paragraph("<b>Next Recommended Steps:</b>", styles['Heading4']),
        Paragraph(
            "• Schedule a physical consultation with a pulmonologist or radiologist.<br/>"
            "• Bring this report and the original image during the visit.<br/>"
            "• Get additional imaging like CT scan or MRI if advised.<br/>"
            "• Maintain personal health logs of symptoms, if any.", styles['Justify']),
        Spacer(1, 10),
        Paragraph("<b>Risk Factors to Watch:</b>", styles['Heading4']),
        Paragraph("• Chronic smoking habits<br/>"
                  "• Family history of lung diseases<br/>"
                  "• Exposure to asbestos, radon, or heavy air pollution<br/>"
                  "• History of lung infections", styles['Justify']),
        Spacer(1, 10),
        Paragraph("<b>Prevention Tips:</b>", styles['Heading4']),
        Paragraph("• Avoid tobacco and smoking<br/>"
                  "• Use air purifiers and avoid heavily polluted environments<br/>"
                  "• Regular exercise and breathing practices<br/>"
                  "• Annual lung health screening for high-risk individuals", styles['Justify']),
        Spacer(1, 10),
        Paragraph("<b>Contact Information:</b>", styles['Heading4']),
        Paragraph("LungScan Diagnostics Center<br/>"
                  "123 Wellness Avenue, MedCity 400012<br/>"
                  "Phone: +91-9876543210 | Email: care@lungscan.in", styles['Justify']),
        Spacer(1, 10),
        Paragraph("<b>Suggested Follow-up:</b>", styles['Heading4']),
        Paragraph("Please consult a specialist within 7 days for further evaluation and possible confirmation tests.", styles['Justify']),
    ]
    frame.addFromList(story, c)

    # Signature/Footer
    c.setFont("Helvetica", 10)
    c.drawString(50, 80, "Authorized Signature: ____________________")
    c.drawString(380, 80, "Contact: +91-9876543210")

    c.save()
    os.remove(temp_img_path)

    speak_async("Report generated and saved")
    update_label("Report generated and saved")


def window():
    root.destroy()


button1 = tk.Button(frame_alpr, text=" Upload Image ", command=openimage, width=15, height=1, font=('times', 15, ' bold '), bg="white", fg="black")
button1.place(x=10, y=40)

button2 = tk.Button(frame_alpr, text="Preprocessing", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '), bg="white", fg="black")
button2.place(x=10, y=100)

button4 = tk.Button(frame_alpr, text="Detection", command=test_model, width=15, height=1, bg="white", fg="black", font=('times', 15, ' bold '))
button4.place(x=10, y=160)

button5 = tk.Button(frame_alpr, text="Generate Report", command=generate_report, width=15, height=1, font=('times', 15, ' bold '), bg="white", fg="black")
button5.place(x=10, y=220)

exit = tk.Button(frame_alpr, text="Logout", command=window, width=15, height=1, font=('times', 15, ' bold '), bg="red", fg="white")
exit.place(x=10, y=270)

root.mainloop()
