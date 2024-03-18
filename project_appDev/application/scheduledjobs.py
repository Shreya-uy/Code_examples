from application.workers import celery
from celery.schedules import crontab
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from application.config import DevConfig
from application.models import *
from sqlalchemy import and_,or_
from datetime import datetime
from jinja2 import Template
from weasyprint import HTML
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from email import encoders
from flask_security import current_user
from sqlalchemy.sql import func
import application.cache as cache
from application.models import User, List, Cards
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import math


@celery.on_after_finalize.connect
def send_daily_scheduled_reminder(sender, **kwargs):
    sender.add_periodic_task(crontab(minute=22,hour=17), daily_emails.s(), name="Everyday at 6.30 pm")

@celery.on_after_finalize.connect
def monthly_pdf_mail(sender,**kwargs):
    sender.add_periodic_task(crontab(day_of_month=14,minute=22,hour=17),send_monthly_pdf.s(), name = "")

# Get list and card 
def get_data(userid=None):
    fulldata_arr = []
    listdata = cache.user_data_alllists(userid)
    listdict = {}

    for li in listdata:
        listdict[li.list_id]=li.list_name

    month = datetime.now().strftime("%Y-%m")
    if month[5:]=="01":
        prevmonth = datetime(int(month[:4]),int(month[5:])+11,1,0,0,0)
        currentmonth = datetime(int(month[:4]),int(month[5:]),1,0,0,0)
    else:
        currentmonth = datetime(int(month[:4]),int(month[5:]),1,0,0,0)
        prevmonth = datetime(int(month[:4]),int(month[5:])-1,1,0,0,0)

    monthdict = {}
    monthdict['Cards created and completed during the month'] = Cards.query.join(List).filter(and_(List.id==userid,func.substr(Cards.created,1,7)==prevmonth.strftime("%Y-%m"),func.substr(Cards.card_due_date,1,7)==prevmonth.strftime("%Y-%m"),Cards.status=="Complete")).all()
    monthdict['Cards created during the month but overdue'] = Cards.query.join(List).filter(and_(List.id==userid,func.substr(Cards.created,1,7)==prevmonth.strftime("%Y-%m"),func.substr(Cards.card_due_date,1,7)==prevmonth.strftime("%Y-%m"),Cards.status=="Incomplete")).all()
    monthdict['Cards created during the month but not due'] = Cards.query.join(List).filter(and_(List.id==userid,func.substr(Cards.created,1,7)==prevmonth.strftime("%Y-%m"),func.substr(Cards.card_due_date,1,7)>prevmonth.strftime("%Y-%m"))).all()
    monthdict['Cards created previously and due in the current month'] = Cards.query.join(List).filter(and_(List.id==userid,func.substr(Cards.created,1,7)<prevmonth.strftime("%Y-%m"),func.substr(Cards.card_due_date,1,7)==prevmonth.strftime("%Y-%m"))).all()
    monthdict['Cards created previously and overdue'] = Cards.query.join(List).filter(and_(List.id==userid,func.substr(Cards.created,1,7)<prevmonth.strftime("%Y-%m"),func.substr(Cards.card_due_date,1,7)<prevmonth.strftime("%Y-%m"))).all()

    for li in listdata:
        li.cards=cache.user_data_cards(userid,li.list_id)
        fulldata_arr.append([li.list_name,li.description,len(li.cards)])
    


    weekly = Cards.query.join(List).filter(and_(List.id==userid,(Cards.completed_date-Cards.card_due_date)<=7,Cards.created>=prevmonth,Cards.created<currentmonth)).count()
    fortnightly = Cards.query.join(List).filter(and_(List.id==userid,(Cards.completed_date-Cards.card_due_date)<=15,(Cards.completed_date-Cards.card_due_date)>7,Cards.created>=prevmonth,Cards.created<currentmonth)).count()
    Othercards = Cards.query.join(List).filter(and_(List.id==userid,(Cards.completed_date-Cards.card_due_date)>15,Cards.created>=prevmonth,Cards.created<currentmonth)).count()
    Incomplete = Cards.query.join(List).filter(and_(List.id==userid,Cards.completed_date==None,Cards.created>=prevmonth,Cards.created<currentmonth)).count()
    
    card_category={}
    card_category['weekly']=weekly
    card_category['fortnightly']=fortnightly
    card_category['others']=Othercards
    card_category['incomplete']=Incomplete
    
    return listdata, card_category, currentmonth, prevmonth, fulldata_arr, monthdict, listdict

# Daily email reminders

@celery.task()
def daily_emails():
    users = User.query.all()
    for user in users:
        listdata = get_data(user.id)[0]
        today = datetime.now().strftime("%Y-%m-%d")
        with open("Reports/Templates/dailyreminder.html") as file:
            template = Template(file.read())
            mailtext = template.render(user=user, listdata = listdata, today = today)
        send_mail(user.email,"Daily Pending Tasks reminder!",mailtext)

# Monthly PDF scheduled
@celery.task
def send_monthly_pdf():
    users = User.query.all()
    month = get_data()[3].strftime("%B-%Y")  
    for user in users:
        with open("Reports/Templates/monthlymail.html") as msg:
                template=Template(msg.read())
                mailtext = template.render(user=user,month=month)
        if (user.report_template=="PDF" or user.report_template ==None):
            generate_pdf(user)
            send_mail(user.email,"Your monthly summary report", mailtext, "Reports/Generated/"+user.Username+"_Monthly PDF Report")
        else:
            report= create_html(user)
            send_mail(to_email=user.email,subject="Your monthly summary report", mail_message = report,attachment="static/"+str(user.id)+"Monthly_cards_plot.png")
        

# Create and Send email function
@celery.task()
def send_mail(to_email, subject,mail_message, attachment=None):
    message = MIMEMultipart()
    message['From'] = DevConfig.SENDER_EMAIL
    message['To'] = to_email
    message['Subject'] = subject
    
    message.attach(MIMEText(mail_message,"html"))
    
    if attachment:
        with open(attachment,"rb") as attachment:
            part=MIMEBase("application","octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",f"attachment; filename = {attachment}")

        message.attach(part)

    server = smtplib.SMTP(host=DevConfig.SMTP_HOST, port=DevConfig.SMTP_PORT)
    server.login(DevConfig.SENDER_EMAIL, DevConfig.SENDER_PASSWORD)
    server.send_message(message)
    server.quit()
    return "Successfully sent daily reminder"

# Generate monthly PDF

@celery.task
def generate_pdf(user):
    monthdata = get_data(user.id)[5]
    prev_month = get_data(user.id)[3].strftime("%B-%Y")
    listdict = get_data(user.id)[6]
    cards_summary(user)
    cardcat = get_data(user.id)[1]
    with open("Reports/Templates/report.html") as file:
        template = Template(file.read())
        report = template.render(user=user, monthdata = monthdata, month = prev_month, card_categories = cardcat, listdict = listdict)
    html=HTML(string=report,base_url="base_url")
    name = user.Username
    report = html.write_pdf(target="Reports/Generated/"+name+"_Monthly PDF Report")
    return ""

@celery.task
def create_html(user):
    monthdata = get_data(user.id)[5]
    prev_month = get_data(user.id)[3].strftime("%B-%Y")
    listdict = get_data(user.id)[6]
    cards_summary(user)
    cardcat = get_data(user.id)[1]
    with open("Reports/Templates/htmlreport.html") as file:
        template = Template(file.read())
        report = template.render(user=user, monthdata = monthdata, month = prev_month, card_categories = cardcat, listdict = listdict)
    return report


# Card chart for monthly PDF

def cards_summary(user):
    cards_due = []
    prevmonth = get_data(user.id)[3]
    total_cards = Cards.query.join(List).filter(and_(List.id==user.id,func.substr(Cards.card_due_date,1,7)<=prevmonth.strftime("%Y-%m"))).count()
    dueandcompleted = Cards.query.join(List).filter(and_(List.id==user.id,func.substr(Cards.card_due_date,1,7)==prevmonth.strftime("%Y-%m"),func.substr(Cards.completed_date,1,7)==prevmonth.strftime("%Y-%m"))).count()
    dueandincomplete = Cards.query.join(List).filter(and_(List.id==user.id,func.substr(Cards.card_due_date,1,7)==prevmonth.strftime("%Y-%m"),or_(func.substr(Cards.completed_date,1,7)>prevmonth.strftime("%Y-%m"),Cards.status=="Incomplete"))).count()
    overdueandcompl = Cards.query.join(List).filter(and_(List.id==user.id,func.substr(Cards.card_due_date,1,7)<prevmonth.strftime("%Y-%m"),func.substr(Cards.completed_date,1,7)==prevmonth.strftime("%Y-%m"))).count()
    overdueandincompl = Cards.query.join(List).filter(and_(List.id==user.id,func.substr(Cards.card_due_date,1,7)<prevmonth.strftime("%Y-%m"),or_(func.substr(Cards.completed_date,1,7)>prevmonth.strftime("%Y-%m"),Cards.status=="Incomplete"))).count()
    
    card_names = []
    if dueandcompleted>0:
        card_names.append("Due and completed during the month")
        cards_due.append(dueandcompleted)
    if dueandincomplete>0:
        card_names.append("Due this month but not completed")
        cards_due.append(dueandincomplete)
    if overdueandcompl>0:
        card_names.append("Overdue from past months and completed")
        cards_due.append(overdueandcompl)
    if overdueandincompl>0:
        card_names.append("Overdue from past months and incomplete")
        cards_due.append(overdueandincompl)
    All_cards_data = pd.DataFrame({'Card':card_names, 'Due status':cards_due})
    sns.set(rc={'figure.figsize':(10,10)})
    fig2 = plt.figure(figsize=(10,10))
    _, _, autotexts = plt.pie(All_cards_data['Due status'], autopct=lambda x: "{a:.0f}/{b:.0f}".format(a=x/100*total_cards,b=total_cards))
    for instance in autotexts:
        instance.set_color('white')
    plt.title("Cards status for the month", fontsize = 18, color = "midnightblue", weight = "bold")
    plt.legend(All_cards_data['Card'], bbox_to_anchor=(0.8, 0.1),prop={'size': 15})
    plt.savefig("static/"+str(user.id)+"Monthly_cards_plot.png")

# Return a csv file

@celery.task
def export_csv_lists(current_user_id):
    alldata = get_data(current_user_id)[4]
    columns = ["Category", "Category description","Card count"]
    df = pd.DataFrame(alldata, columns = columns)
    df.to_csv("Reports/Generated/"+str(current_user_id)+"_lists.csv",index=False)
    return "Success"

@celery.task
def export_csv_cards(current_user_id,listid):
    alldata = cache.user_data_cards(current_user_id,listid)
    cards = []
    for card in alldata:
        cards.append([card.card_title,card.card_content,card.created,card.card_due_date,card.completed_date,card.last_modified])
    columns = ["Task name", "Task description","Task created on", "Task due date","Task completed on","Task last modified"]
    df = pd.DataFrame(cards, columns = columns)
    df.to_csv("Reports/Generated/"+str(current_user_id)+"_"+listid+"_cards.csv",index=False)
    return "Success"
    
@celery.task
def import_csv(userid):
    with open("Reports/Uploaded/"+str(userid)+".csv") as file:
        data = pd.read_csv(file)
    alllists = cache.user_data_alllists(userid)
    listdict = dict(map(lambda x:(x.list_name,x.list_id), alllists))
    for i in range(len(data)):
        row = data.iloc[i]
        if row["Category name"] in listdict:
            fetchedlist = List.query.filter(List.list_id==listdict[row['Category name']]).first()
            cards = cache.user_data_cards(userid,fetchedlist.list_id)
            cardnames = list(map(lambda x:x.card_title, cards))
            cardexists = False
            listadd = False
            for cardname in cardnames:
                if cardname == row["Task name"]:
                    cardexists = True
                    break
            insertdb(userid,listdict,row,cardexists,listadd)
            cache.delete_cache_cards(userid,listdict[row["Category name"]])
        elif row["Category name"]!="":
            listadd=True
            listid = insertdb(userid=userid,row=row,listadd=listadd)
            listdict[row["Category name"]]=listid
            listadd=False
            insertdb(userid=userid,list=listdict,row=row,listadd=listadd)
        cache.delete_cache_alllists(userid)
        cache.delete_cache_cards(userid)
    return ""

engine = create_engine("sqlite:///database/Proj_db.sqlite3")

def insertdb(userid,list=None,row=None,cardexists=False,listadd=False):
    session = Session(engine, autoflush = False)
    session.begin()
    res = checkdate(row)
    if listadd==False and cardexists and res==True:
        fetchedcard = Cards.query.join(List).filter(and_(List.id==userid,List.list_id==list[row["Category name"]],
        Cards.card_title==row["Task name"])).first()
        if res == True:
            fetchedcard.card_title = row["Task name"]
            fetchedcard.card_content = row["Task description"]
            fetchedcard.created = row["Task created on"]
            fetchedcard.card_due_date = row["Task due date"]
            fetchedcard.completed_date = row["Task completed on"]
            if str(row["Task modified on"])=="nan":
                fetchedcard.last_modified = row["Task created on"]
            else:
                fetchedcard.last_modified = row["Task modified on"]
    elif listadd==False and (str(row["Task name"])!="nan" and res==True):
        listinstance = session.query(List).filter(and_(List.id==userid,List.list_id==list[row["Category name"]])).first()
        if str(row["Task modified on"])=="nan":
            modifieddate = row["Task created on"]
        else:
            modifieddate = row["Task modified on"]
        entry = Cards(list_id=listinstance.list_id,card_title=row["Task name"],card_content = row["Task description"],created=row["Task created on"],
        card_due_date = row["Task due date"],completed_date = row["Task completed on"],last_modified=modifieddate)
        session.add(entry)
    elif listadd==True:
        listentry = List(id=userid,list_name=row["Category name"],description=row["Category description"])
        session.add(listentry)
    session.flush()
    session.commit()
    session.close()
    listid = List.query.filter(and_(List.id==userid,List.list_name==row["Category name"])).first().list_id
    return listid

def checkdate(row):
    createddate = str(row["Task created on"])
    card_due_date = str(row["Task due date"])
    cardcompleted = str(row["Task completed on"])
    cardmodified = str(row["Task modified on"])
    formatdate = '%Y-%m-%d'
    try:
        res = bool(datetime.strptime(createddate, formatdate))
        res = bool(datetime.strptime(card_due_date, formatdate))
        if cardcompleted !="nan":
            res = bool(datetime.strptime(cardcompleted, formatdate))
        if cardmodified !="nan":
            res = bool(datetime.strptime(cardmodified, formatdate))
    except ValueError:
        res = False
    return res
