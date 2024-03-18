from flask_restful import Resource, Api
from flask_restful import fields, marshal_with
from flask_restful import reqparse
from application.models import User,List,Cards
from application.database import db
from flask import current_app as app
from flask_security import current_user,auth_required
from flask import request
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from flask import flash
from sqlalchemy import and_
from datetime import datetime
from flask import send_file
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import application.cache as cache
from application import scheduledjobs
from flask_security import hash_password 
import werkzeug


user_parser=reqparse.RequestParser()
user_parser.add_argument('Username')
user_parser.add_argument('password')
user_parser.add_argument('report_template')

list_parser = reqparse.RequestParser()
list_parser.add_argument('listname')
list_parser.add_argument('listdesc')


card_parser = reqparse.RequestParser()
card_parser.add_argument('cardname')
card_parser.add_argument('carddesc')
card_parser.add_argument('carddue')

file_parser = reqparse.RequestParser()
file_parser.add_argument('file',location='files')

engine = create_engine("sqlite:///database/Proj_db.sqlite3")


user_fields = {
    'Username':   fields.String,
    'email':   fields.String,
    'password': fields.String,
    'report_template': fields.String
}

list_fields = {
    'list_id':   fields.Integer,
    'description':    fields.String,
    'list_name':    fields.String,
    'id':    fields.String
}

card_fields = {
    'card_title':  fields.String,
    'list_id':    fields.Integer,
    'card_content':    fields.String,
    'created':    fields.String,
    'card_due_date':    fields.String,
    'status':    fields.String,
    'completed_date':    fields.String,
    'last_modified': fields.String
}

class UserAPI(Resource):

    @auth_required('token')
    @marshal_with(user_fields)
    def get(self):
        return current_user

    @auth_required('token')
    def post(self):
        args=user_parser.parse_args()
        username = args.get('Username',None)
        password = args.get('password',None)
        reporttemplate = args.get('report_template',None)
        session = Session(engine, autoflush = False)
        session.begin()
        user = session.query(User).filter(User.id==current_user.id).first()
        if username!="":
            user.Username = username
        if password!="":
            user.password = hash_password(password)
        if reporttemplate!=None:
            user.report_template = reporttemplate
        session.flush()
        session.commit()
        session.close()
        user = session.query(User).filter(User.id==current_user.id).first()
        return {'message':'User successfully added','username':user.Username}

class ListAPI(Resource):
    
    @auth_required('token')
    @marshal_with(list_fields)
    def get(self):
        session = Session(engine, autoflush = False)
        session.begin()
        lists=cache.user_data_alllists(current_user.id)
        return lists

    @auth_required('token')
    def post(self):
        args = list_parser.parse_args()
        list_name = args.get("listname",None).strip()
        list_desc = args.get("listdesc",None)
        session = Session(engine, autoflush = False)
        session.begin()
        entry = List(list_name = list_name, id = current_user.id, description = list_desc)
        lists = cache.user_data_alllists(current_user.id)
        for list in lists:
            if list_name == list.list_name: 
                return {'message':'failure','listname':entry.list_name, 'description':entry.description,'listid':entry.list_id}   
        cache.delete_cache_alllists(current_user.id)
        session.add(entry)
        session.commit()
        return {'message':'success','listname':entry.list_name,'listid':entry.list_id,'description':entry.description}

    @auth_required('token')
    def put(self,listid):
        args = list_parser.parse_args()
        elist_name = args.get("listname",None).strip()
        edescription = args.get("listdesc",None)
        session = Session(engine, autoflush = False, expire_on_commit=False)
        session.begin()
        item = session.query(List).filter(and_(List.list_user==current_user,List.list_id == listid)).one()
        flag = session.query(List).filter(and_(List.list_user == current_user, List.list_name == elist_name)).first()
        if flag and int(listid) != flag.list_id:
            return {'message':'failure','listname':item.list_name, 'description': item.description}
        else:
            cache.delete_cache_alllists(current_user.id)
            item.list_name = elist_name
            item.description = edescription
            session.flush()
            session.commit()
            session.close()
            return {'message':'edited list','listname':item.list_name,'description':item.description}
    
    @auth_required('token')
    def delete(self,listid):
        session = Session(engine, autoflush = False)
        session.begin()
        entry = session.query(List).filter(and_(List.list_user==current_user,List.list_id == listid)).one()
        for card in entry.list_card:
            session.delete(card)
        cache.delete_cache_alllists(current_user.id)
        cache.delete_cache_cards(current_user.id,listid)
        session.delete(entry)
        session.flush()
        session.commit()
        session.close()
     


class CardAPI(Resource):
    
    @auth_required('token')   
    @marshal_with(card_fields)
    def get(self,listid=None):
        if listid==None:
            cards=cache.user_data_allcards(current_user.id)
            return cards
        else:
            cards=cache.user_data_cards(current_user.id,listid)
            return cards
    
    @marshal_with(card_fields)
    @auth_required('token')
    def post(self,listid):
        args = card_parser.parse_args()
        cname = args.get("cardname",None).rstrip()
        ccontent = args.get("carddesc",None)
        cdate = args.get("carddue",None)
        entry = Cards(card_title = cname, card_content = ccontent, card_due_date = cdate, list_id = listid)
        session = Session(engine, autoflush = False, expire_on_commit=False)
        session.begin()
        cards = cache.user_data_cards(current_user.id,listid)
        for card in cards:
            if card.card_title == cname:
                return {'message':'failure'}
        cache.delete_cache_cards(current_user.id,listid)
        session.add(entry)
        session.flush()
        session.commit()
        session.close()
        newcard=Cards.query.join(List).filter(and_(List.list_user==current_user,Cards.list_id==listid,Cards.card_title==cname)).first()
        return newcard

    @auth_required('token')
    def put(self,listid,cardtitle):
        args = card_parser.parse_args()
        cname = args.get("cardname",None).rstrip()
        ccontent = args.get("carddesc",None)
        cdate = args.get("carddue",None)
        session = Session(engine, autoflush = False)
        session.begin()
        card = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid,Cards.card_title == cardtitle)).one()
        card_names = list(map(lambda x:x[0].strip('\r\n'),session.query(Cards.card_title).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid,Cards.card_title!=cardtitle)).all())) 
        if cname in card_names: 
            return {'message':'failure','cardtitle':card.card_title}
        else:
            card.card_title = cname
            card.card_content = ccontent
            card.card_due_date = cdate
            card.last_modified = datetime.now()
            session.flush()
            session.commit()
            cache.delete_cache_cards(current_user.id,listid)
            return {'message':'card edited'}

    @auth_required('token')
    def delete(self,listid,cardtitle):
        session = Session(engine, autoflush = False)
        session.begin()
        entry = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid,Cards.card_title == cardtitle)).one()
        session.delete(entry)
        session.flush()
        session.commit()
        session.close()
        cache.delete_cache_cards(current_user.id,listid)
        return {'message':'Card' + str(cardtitle)+'deleted'}

    @auth_required('token')
    def patch(self,listid, cardtitle,newlistid=None):
        due_date = request.form.get('carddue',None)
        session = Session(engine, autoflush = False)
        session.begin()
        flag = True
        cards = cache.user_data_cards(current_user.id,newlistid)
        if newlistid:
            for card in cards:
                if card.card_title == cardtitle:
                    flag = False
            if flag == True:
                card = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid, Cards.card_title == cardtitle)).one()
                card.list_id = newlistid
            elif newlistid==listid:
                return {'message':'samelist'}
            else:
                return {'message':'failure'}
        card = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid, Cards.card_title == cardtitle)).one()
        if due_date:
            if due_date>=datetime.now().strftime("%Y-%m-%d"):
                card = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid, Cards.card_title == cardtitle)).one()
                card.card_due_date = due_date
        if newlistid==None and due_date==None:
            if card.status == "Incomplete":            
                card.status = "Complete"
                card.completed_date = datetime.now().strftime("%Y-%m-%d")
            elif card.status == "Complete":
                card.status = "Incomplete"
                card.completed_date = None
        cache.delete_cache_cards(current_user.id,listid)
        cache.delete_cache_cards(current_user.id)
        session.flush()
        session.commit()
        session.close()
        if newlistid==None:
            card = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == listid, Cards.card_title == cardtitle)).first()
        else:
            card = session.query(Cards).join(List).filter(and_(List.list_user==current_user,Cards.list_id == newlistid, Cards.card_title == cardtitle)).first()
        return {'message':'success','card_status':card.status,'card_due_date':card.card_due_date,'completed_date':card.completed_date}


class ExportAPI(Resource):
    
    @auth_required('token')
    def get(self):
        current_user_id = current_user.id
        job = scheduledjobs.export_csv_lists.delay(current_user_id)
        job.wait()
        return send_file("Reports/Generated/"+str(current_user_id)+"_lists.csv", as_attachment=True, download_name = 'Dashboard report')

    @auth_required('token')
    def post(self):
        listid = request.form.get('listid')
        current_user_id = current_user.id
        job = scheduledjobs.export_csv_cards.delay(current_user_id,listid)
        job.wait()
        return send_file("Reports/Generated/"+str(current_user_id)+"_"+listid+"_cards.csv", as_attachment=True, download_name = 'Dashboard report')
        
class ImportAPI(Resource):

    @auth_required('token')
    def get(self):
        return send_file("Reports/Files_to_upload/Sample report format.csv", as_attachment = True)

    @auth_required('token')
    def post(self):
        csvfile = request.files['file']
        csvfile.save("Reports/Uploaded/"+str(current_user.id)+".csv")
        job = scheduledjobs.import_csv.delay(current_user.id)
        job.wait()
        return {'message':'File successfully uploaded'}
        

class ListSummaryAPI(Resource):
    
    @auth_required('token')
    def get(self):
        session = Session(engine, autoflush=False)
        session.begin()
        all_lists = cache.user_data_alllists(current_user.id)
        cards_duelater, cards_duenow, cards_overdue, list_names = [],[],[],[]
        for list in all_lists:
            cd = datetime.now().strftime("%Y-%m-%d")
            list_names.append(list.list_name)
            duecount1 = session.query(Cards).filter(Cards.list_id == list.list_id).filter(Cards.card_due_date>cd).count()
            duecount2 = session.query(Cards).filter(Cards.list_id == list.list_id).filter(Cards.card_due_date==cd).count()
            duecount3 = session.query(Cards).filter(Cards.list_id == list.list_id).filter(Cards.card_due_date<cd).count()
            cards_duelater.append(duecount1)
            cards_duenow.append(duecount2)
            cards_overdue.append(duecount3)
        All_lists_data = pd.DataFrame({'List':list_names, 'Due later':cards_duelater,'Due today':cards_duenow,'Overdue':cards_overdue})
        sns.set_palette("viridis")
        fig1 = plt.figure(figsize=(30,30))
        All_lists_plot = All_lists_data.set_index('List').plot(kind='bar',stacked = True)
        plt.xlabel("List", fontsize = 12)
        plt.ylabel("Count of lists", fontsize = 12)
        plt.title("Card status by list", fontsize = 18, weight = "bold", color = "Teal")
        ylabels = []
        for y in All_lists_plot.get_yticks():
            if int(y)==y:
                ylabels.append(int(y))
            else:
                ylabels.append("")
        All_lists_plot.set_yticklabels(ylabels)
        All_lists_plot.set_xticklabels(All_lists_plot.get_xmajorticklabels(),rotation=45)
        figure_1 = All_lists_plot.get_figure()
        figure_1.subplots_adjust(bottom=0.32)
        figure_1.savefig("static/All_lists_plot.png")
        session.close()
        return send_file("static/All_lists_plot.png")

class CardSummaryAPI(Resource):

    @auth_required('token')
    def get(self):
        session = Session(engine, autoflush=False)
        session.begin()
        lists = list(map(lambda x:x.list_id, cache.user_data_alllists(current_user.id)))
        total_cards = session.query(Cards).filter(Cards.list_id.in_(lists)).count()
        cards_due = []
        cd = datetime.now().strftime("%Y-%m-%d")
        duecount1 = session.query(Cards).filter(Cards.list_id.in_(lists)).filter(Cards.card_due_date>cd).count()
        duecount2 = session.query(Cards).filter(Cards.list_id.in_(lists)).filter(Cards.card_due_date==cd).count()
        duecount3 = session.query(Cards).filter(Cards.list_id.in_(lists)).filter(Cards.card_due_date<cd).count()
        card_names = []
        if duecount1>0:
            card_names.append("Due later")
            cards_due.append(duecount1)
        if duecount2>0:
            card_names.append("Due today")
            cards_due.append(duecount2)
        if duecount3>0:
            card_names.append("Overdue")
            cards_due.append(duecount3)
        All_cards_data = pd.DataFrame({'Card':card_names, 'Due status':cards_due})
        sns.set_palette("rocket")
        fig2 = plt.figure(figsize=(5,5))
        _, _, autotexts = plt.pie(All_cards_data['Due status'], labels = All_cards_data['Card'], autopct=lambda x: "{a:.0f}/{b:.0f}".format(a=x/100*total_cards,b=total_cards))
        for instance in autotexts:
            instance.set_color('white')
        plt.title("Overall cards status", fontsize = 18, color = "Purple", weight = "bold")
        plt.savefig("static/All_cards_plot.png")
        session.close()
        return send_file("static/All_cards_plot.png")

class CompletionSummaryAPI(Resource):

    @auth_required('token')
    def get(self):
        session = Session(engine, autoflush=False)
        session.begin()
        lists = list(map(lambda x:x.list_id, cache.user_data_alllists(current_user.id)))
        cards_completed = list(map(lambda x:x[0],session.query(Cards.completed_date).filter(Cards.list_id.in_(lists)).filter(Cards.status == "Complete").all()))
        cards_completed_data = pd.DataFrame({'Completed dates':cards_completed,'Count':cards_completed})
        due_dates = cards_completed_data.groupby(['Completed dates']).count()
        set = ["No of cards" for x in due_dates.Count]
        if not due_dates.empty:
            sns.set_palette("rocket_r")
            fig2 = plt.figure(figsize=(6,6))
            plot = sns.lineplot(data = due_dates, x= 'Completed dates', y = 'Count', style = set, markers = ['o'])
            plt.ylim(0, max(due_dates['Count'])+1)
            plot.get_legend().legendHandles[0].set_color('orange')
            ylabels = []
            for y in plot.get_yticks():
                if int(y)==y:
                    ylabels.append(int(y))
                else:
                    ylabels.append("")
            plot.set_yticklabels(ylabels)
            plt.title("Card completion timeline", fontsize = 18, color = "Orange", weight = "bold")
            plot.set_xticklabels(plot.get_xmajorticklabels(),rotation=40)
            plt.subplots_adjust(bottom=0.32)
            plt.savefig("static/Completion_timeline_plot.png")
            session.close()
        else:
            return {"message":"No cards completed yet"}
        return send_file("static/Completion_timeline_plot.png")

class LastUpdateAPI(Resource):

    @auth_required('token')
    def get(self):
        lastupdate = {}
        all_lists = cache.user_data_alllists(current_user.id)
        
        for list in all_lists:
            cards = Cards.query.join(List).filter(and_(List.id==current_user.id,List.list_id==list.list_id)).all()
            maxdate = None
            for card in cards:
                if maxdate == None:
                    maxdate = card.completed_date
                else:
                    if card.completed_date!=None and card.completed_date>maxdate:
                        maxdate = card.completed_date
            lastupdate[list.list_name]=maxdate
        return json.dumps(lastupdate)

