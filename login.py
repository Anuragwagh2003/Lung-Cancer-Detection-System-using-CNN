import tkinter as tk
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import pyttsx3
import threading
  # Importing pyttsx3 for speech functionality

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Function to speak the text
def speak(text):
    engine.say(text)
    engine.runAndWait()
def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()

# Initialize the Tkinter window
root = tk.Tk()
root.configure(background="white")
root.geometry("600x600")
root.title("Login Form")

# Variables for username and password
username = tk.StringVar()
password = tk.StringVar()

# Registration function
def registration():
    from subprocess import call
    call(["python", "registration.py"])
    root.destroy()

# Login function with speech and message display
def login():
    # Establish Connection with SQLite Database
    with sqlite3.connect('evaluation.db') as db:
        c = db.cursor()

        # Creating the table if it doesn't exist
        c.execute("CREATE TABLE IF NOT EXISTS admin_registration"
                  "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT, Gender TEXT, age TEXT, password TEXT)")
        db.commit()

        # Check if the username and password match
        find_entry = ('SELECT * FROM admin_registration WHERE username = ? and password = ?')
        c.execute(find_entry, [(username.get()), (password.get())])
        result = c.fetchall()

        if result:
            msg = "Login successfully"
          
            speak_async(msg) 
            ms.showinfo("Message", msg)
              # Speak the success message
            
            # Store the logged-in username to a file
            with open("logged_in_user.txt", "w") as f:
                f.write(username.get())

            root.destroy()
            from subprocess import call
            call(['python', 'GUI_Master_old.py'])

        else:
            msg = "Oops! Username or Password did not match"
            speak_async(msg)
            ms.showerror('Error', msg)
             # Speak the error message

# UI Components
title = tk.Label(root, text="Login Here", font=("Algerian", 30, "bold", "italic"), bd=5, bg="black", fg="white")
title.place(x=200, y=100, width=250)

Login_frame = tk.Frame(root, bg="white")
Login_frame.place(x=100, y=150)

logolbl = tk.Label(Login_frame, bd=0).grid(row=0, columnspan=2, pady=20)

lbluser = tk.Label(Login_frame, text="Username", font=("Times new roman", 20, "bold"), bg="white").grid(row=1, column=0, padx=20, pady=10)
txtuser = tk.Entry(Login_frame, bd=5, textvariable=username, font=("", 15))
txtuser.grid(row=1, column=1, padx=20)

lblpass = tk.Label(Login_frame, text="Password", font=("Times new roman", 20, "bold"), bg="white").grid(row=2, column=0, padx=50, pady=10)
txtpass = tk.Entry(Login_frame, bd=5, textvariable=password, show="*", font=("", 15))
txtpass.grid(row=2, column=1, padx=20)

btn_log = tk.Button(Login_frame, text="Login", command=login, width=15, font=("Times new roman", 14, "bold"), bg="Green", fg="black")
btn_log.grid(row=3, column=1, pady=10)

btn_reg = tk.Button(Login_frame, text="Create Account", command=registration, width=15, font=("Times new roman", 14, "bold"), bg="red", fg="black")
btn_reg.grid(row=3, column=0, pady=10)

root.mainloop()
