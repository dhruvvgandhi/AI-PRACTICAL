#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pyttsx3
from datetime import datetime


def greet(nm):
    now=datetime.now()
    hours=int(now.strftime("%H"))
    if hours in range(4,13):
        s="Good Morning "+ nm +"!"
    elif hours in range(13,17):
        s="Good Afternoon "+ nm +"!"
    else:
        s="Good Evening "+ nm +"!"
    return s
        

os.system("cls")

print()

print("\t\t\t\t Hey! I am Anik, your system's personal Assistant")
print("\t\t\t\t -------------------------------------------------")
pyttsx3.speak("Hey! I am Anik, your system's personal Assistant")

print()
print("\t\t\t\t Welcome to Anik's World (Menu Based Service BOT) \t\t\t")
print("\t\t\t\t -------------------------------------------------")
pyttsx3.speak("Welcome to Anik's World (Menu Based Service BOT)")

print()

print("I have introduced myself, now it's time for me to know you ")
pyttsx3.speak("I have introduced myself, now it's time for me to know you ")
print("Please write your name here :", end=" ")
pyttsx3.speak("Please write your name here :")
name=input()

greetings=greet(name)
print(greetings)
pyttsx3.speak(greetings)
print()
print("\t\t Here's the list of tasks that I can perform for you")
print("\t\t ...................................................")
pyttsx3.speak("Here's the list of tasks that I can perform for you :")
print()
print('''
         >>Run Applications:
               -Google chrome
               -Mozila Firfox
               -calculator
               -calender
               -notepad
               -Alarms and Clock
               -Camera
               -Cortana
               -Microsoft Edge
               -Microsoft Store
               -Photos
               -Screen Snip
               -VS code
               -Jupyter notebook
               -VLC media Player
               -Windows Media Player
               -MS Word
               -MS PowerPoint
               -Ms Excel
               
         >>Open Web-page:
               -Facebook
               -Gmail
               -Twitter
               -Github
               -YouTube
               -LinkedIn
               
               ''')

print()



hl="How Can I help you "+name+" ?"
print(hl)
pyttsx3.speak(hl)

while(True):
    c=input("Command Me: ")
    c=c.lower()
    
    if "do not" in c or "don't" in c or "dont" in c or "donot" in c:
        print("Cool!! I won't run this")
        pyttsx3.speak("Cool!! I won't run this")
        print()
        
    else:
        if "Google Chrome" in c:
            print("Opening Chrome")
            pyttsx3.speak("Opening Chrome")
            os.system("Google Chrome")
            print()
        elif("firefox" in c or "mozila" in c):
            print("Opening Mozila Firefox")
            pyttsx3.speak("Opening Mozila Firefox")
            os.system("Google Chrome")
            print()
        elif("calc" in c or "calculator" in c):
            print("Opening Calculator")
            pyttsx3.speak("Opening Calculator")
            os.system("Start calculator:")  
            print()
        
        elif("calender" in c  ):
            print("Opening Calender")
            pyttsx3.speak("Opening Calender")
            os.system("Start outlookcal:")  
            print()    
        elif("notepad" in c or "editor" in c ):
            print("Opening NotePad")
            pyttsx3.speak("Opening NotePad")
            os.system("NotePad")  
            print()
        elif("clock" in c or "alarms" in c ):
            print("Opening Alarms and Clock")
            pyttsx3.speak("Opening Alarms and Clock")
            os.system("Start ms-clock:")  
            print()
        elif("camera" in c or "cam" in c ):
            print("Opening Camera")
            pyttsx3.speak("Opening Camera")
            os.system("Start microsoft.windows.camera:")  
            print()
        elif("cortana" in c  ):
            print("Opening Cortana")
            pyttsx3.speak("Opening Cortana")
            os.system("Start ms-cortana:")  
            print()
        elif("edge" in c  ):
            print("Opening Microsoft Edge")
            pyttsx3.speak("Opening Microsoft Edge")
            os.system("Start microsoft-edge:")
            print()
        elif("store" in c  ):
            print("Opening Microsoft Store")
            pyttsx3.speak("Opening Microsoft Store")
            os.system("Start ms-windows-store:")
            print()
        elif("photos" in c  ):
            print("Opening Photos")
            pyttsx3.speak("Opening Photos")
            os.system("Start ms-photos:")
            print()   
         
        elif("snip" in c  ):
            print("Opening Screen-Snip")
            pyttsx3.speak("Opening Screen-Snip")
            os.system("Start ms-screenclip:")
            print()   
        elif("word" in c  ):
            print("Opening MS-Word")
            pyttsx3.speak("Opening MS-Word")
            os.system("winword")
            print()
        elif("excel" in c  ):
            print("Opening MS-Excel")
            pyttsx3.speak("Opening MS-Excel")
            os.system("excel")
            print()
        elif("powerpoint" in c or "power point" in c or "ppt" in c ):
            print("Opening MS-PowerPoint")
            pyttsx3.speak("Opening MS-PowerPoint")
            os.system("powerpnt")
            print()
        elif("vscode" in c or "vs-code" in c or "code" in c or "visual studio code" in c or "visualstudiocode" in c):
            print("Opening VS code ")
            pyttsx3.speak("Opening VS code")
            os.system("code")
            print()
        elif("vlc" in c ):
            print("Opening Vlc Media Player")
            pyttsx3.speak("Opening Vlc Media Player")
            os.system("vlc")
            print()
        elif("windows media player" in c or "wm player" in c ):
            print("Opening Windows media player")
            pyttsx3.speak("Opening Windows media player")
            os.system("wmplayer")
            print()
        elif("jupyter" in c ):
            print("Opening Jupyter Notebook")
            pyttsx3.speak("Opening Jupyter Notebook")
            os.system("jupyter notebook")
            print()
            
        elif ("youtube") in c:
            print( "Opening YouTube" )
            pyttsx3.speak("Opening Youtube")
            os.system("chrome youtube.com")
            print()
        elif ("facebook" in c or "fb" in c ):
            print( "Opening Facebook" )
            pyttsx3.speak("Opening Facebook")
            os.system("chrome facebook.com") 
            print()
        elif ("gmail" in c or "mail" in c ):
            print( "Opening Gmail" )
            pyttsx3.speak("Opening Gmail")
            os.system("chrome gmail.com")
            print()
        elif ("twitter" in c):
            print( "Opening Twitter" )
            pyttsx3.speak("Opening Twitter")
            os.system("chrome twitter.com") 
            print()
        elif ("github" in c  ):
            print( "Opening GitHub" )
            pyttsx3.speak("Opening gitHub")
            os.system("chrome github.com") 
            print()
        elif ("linkedin" in c  ):
            print( "Opening Linkedin" )
            pyttsx3.speak("Opening Linkedin")
            os.system("linkedin.com") 
            print()
        elif ("thanks" in c or "ty" in c or "thankyou" in c or "thank you" in c  ):
            print( "Always there to help u" )
            pyttsx3.speak("Always there to help u")
            print()
        
        elif "bye" in c or "quit" in c or "exit" in c:
            print('''Bye!! See You Soon
                     It was a nice time with you''')
            pyttsx3.speak("Bye!! See You Soon.....It was a nice time with you")
            exit() 
        
        else:
            print( "Can't Help you in this!! Will try to help u regarding this, in future" )
            pyttsx3.speak("Can't Help you in this!!  Will try to help u regarding this, in future")


# In[ ]:




