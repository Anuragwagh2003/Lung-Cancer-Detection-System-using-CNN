import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import sqlite3
import pyttsx3 
import threading
#import tfModel_test as tf_test
global fn
fn=""

engine = pyttsx3.init()

# Function to speak the text
def speak(text):
    engine.say(text)
    engine.runAndWait()

##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Lung cancer disease Detection using CNN")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('l2.jpeg')
image2 =image2.resize((900,700), Image.LANCZOS)

background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image,bd=5)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)
#


#frame_display = tk.LabelFrame(root, text=" --Display-- ", width=900, height=250, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display.grid(row=0, column=0, sticky='nw')
#frame_display.place(x=300, y=100)

#frame_display1 = tk.LabelFrame(root, text=" --Result-- ", width=900, height=200, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display1.grid(row=0, column=0, sticky='nw')
#frame_display1.place(x=300, y=430)

#frame_display2 = tk.LabelFrame(root, text=" --Calaries-- ", width=900, height=50, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display2.grid(row=0, column=0, sticky='nw')
#frame_display2.place(x=300, y=380)

frame_alpr = tk.LabelFrame(root, text="  ", width=900, height=1500, bd=5, font=('times', 14, ' bold '),bg="#271983")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=900, y=0)

lbl = tk.Label(root, text="Lung Cancer Disease Detection using CNN", font=('Elephant', 28,' bold '),width=50,bg="blue",fg="white")
lbl.place(x=0, y=0)

lbl = tk.Label(frame_alpr, text='Cancer is a part of our life ', font=('Lucida Calligraphy', 15,' bold '),bg="#271983",fg="white")
lbl.place(x=30, y=100)

lbl = tk.Label(frame_alpr, text="but it's not our whole life", font=('Lucida Calligraphy', 15,' bold '),bg="#271983",fg="white")
lbl.place(x=30, y=140)




def login():
    speak("You are about to log in. Please enter your username and password.")
    from subprocess import call
    call(["python", "login.py"])  

def register():
    speak("You are about to sign up. Please provide your details to create an account.")
    from subprocess import call
    call(["python", "registration.py"])  
   
def window():
    root.destroy()

button1 = tk.Button(frame_alpr, text=" SIGN UP ",command=register,width=15, height=1, font=('times', 15, ' bold '),bg="green",fg="white")
button1.place(x=100, y=350)

button2 = tk.Button(frame_alpr, text="LOGIN",command=login,width=15, height=1, font=('times', 15, ' bold '),bg="blue",fg="white")
button2.place(x=100, y=450)




root.mainloop()