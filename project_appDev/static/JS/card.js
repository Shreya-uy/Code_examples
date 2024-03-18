import {tick, pencil_square, arrow_left_right,del_icon} from "./svg.js"
var card = {data: function(){return {tick: tick, pencil_square:pencil_square, alr:arrow_left_right,del_icon:del_icon,color:"",
        current_date:this.revformatdate(new Date()),temp:{}}},
            props:['card','list'],
            methods:{
                change_color: function(duedate,cardstatus,completeddate){
                    if (cardstatus=="Incomplete"){
                    if (duedate>this.current_date){
                        this.color="text-success"
                    }
                    else if (duedate==this.current_date){
                        this.color="text-yellow"
                    }
                    else {
                        this.color="text-danger"
                    }
                    return this.color
                }
                else 
                {if (duedate>completeddate){
                    this.color="text-success"
                }
                else if (duedate==completeddate){
                    this.color="text-yellow"
                }
                else {
                    this.color="text-danger"
                }
                return this.color}
            },
            cardChangename (event) {
                let obj = {}
                let inputs = event.target.id.split(",")
                let lid = inputs[0]
                let cardtitleold=inputs[1]
                obj['cardname']=event.target.textContent
                obj['carddesc']=this.card.card_content
                obj['carddue']=this.card.card_due_date
                let edit_obj = JSON.stringify(obj)

                fetch("/api/cards/"+lid+"/"+cardtitleold,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"PUT",body:edit_obj})
                .then(resp=>resp.json())
                .then(data=> {if (data.message=="failure"){event.target.innerText=data.cardtitle; this.card['card_title']=data.cardtitle; alert("Card already exists in this list, use a different name")}
                else{this.card['card_title']=event.target.textContent}})
            },

            cardChangecontent (event) {
                let obj = {}
                let inputs = event.target.id.split(",")
                let lid = inputs[0]
                let cardtitleold=inputs[1]
                obj['cardname']=this.card.card_title
                obj['carddesc']=event.target.textContent
                obj['carddue']=this.card.card_due_date
                let edit_obj = JSON.stringify(obj)

                fetch("/api/cards/"+lid+"/"+cardtitleold,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"PUT",body:edit_obj})
                .then(resp=>resp.json())
                .then(data=> {this.card['card_content']=event.target.textContent})
            },
            checkchar(event){
                if (event.target.textContent.length>150){
                    alert("Please enter 150 characters or less")
                }
            },
            async deletecard(listid,cardtitle){
                await fetch("/api/cards/"+listid+"/"+cardtitle,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"DELETE"})
                .then(res=>res.json())
                .then(async data=>{
                await fetch("/api/cards",{'headers':{'Authentication-Token':localStorage.getItem('token')}})
                .then(response=> response.json())
                .then(data=>
                {data.forEach(card=>
                {   if (!(card.list_id in this.temp)){
                    this.temp[card.list_id]=[]
                }
                this.temp[card.list_id].push(card)}),this.$store.commit('setcard',this.temp)})
                .catch(err=>console.log(err))
                })
            },
            mark_complete(event){
                let params=event.target.parentElement.parentElement.id.split(',')
                let listid=params[0]
                let cardname=params[1]
                fetch("/api/cards/"+listid+"/"+cardname,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"PATCH"})
                .then(resp=>resp.json())
                .then(data=> {this.card['status']=data.card_status})
              },
              mark_incomplete(event){
                let params=event.target.id.split(',')
                let listid=params[0]
                let cardname=params[1]
                fetch("/api/cards/"+listid+"/"+cardname,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"PATCH"})
                .then(resp=>resp.json())
                .then(data=> {this.card['status']=data.card_status})
              },
              formatdate(date){
                let datemod =  new Date(date)
                let datemonth =  datemod.toLocaleDateString('en-us', {year:"numeric", month:"short", day:"numeric"}).split(',')[0]
                let year = datemod.toLocaleDateString('en-us', {year:"numeric", month:"short", day:"numeric"}).split(',')[1].substring(3,5)
                let day = datemonth.split(' ')[1]
                let month = datemonth.split(' ')[0]
                return day+"-"+month+"-"+year
              },
              revformatdate(date){
                let datestr = new Date(date).toLocaleDateString({year:"numeric", month:"numeric", day:"numeric"})
                let day = datestr.split('/')[0]
                let month = datestr.split('/')[1]
                let year = datestr.split('/')[2]
                let finaldate = year+"-"+month+"-"+day
                return finaldate
              },
              cardchangedate(event){
                let params=event.target.id.split(',')
                let listid=params[0]
                let cardname=params[1]
                let newdate = this.revformatdate(event.target.textContent.split(' ')[1])
                if (newdate.includes("Invalid")){
                    alert("Please enter valid date")
                    event.target.innerText = "Due: "+this.formatdate(this.card.card_due_date)
                }
                else{
                let obj = new FormData()
                obj.append('carddue',newdate)
                fetch("/api/cards/"+listid+"/"+cardname,{headers:{'Authentication-Token':localStorage.getItem('token')},method:"PATCH",body:obj})
                .then(res=>res.json())
                .then((data)=>{
                    this.card.card_due_date=data.card_due_date,
                    event.target.innerText = "Due: "+this.formatdate(this.card.card_due_date),
                    event.target.classList=[],
                    event.target.classList.add(this.change_color(data.card_due_date,data.card_status,data.completed_date))})
                }
              }
            },

            template: `	<div class="shadow card mb-3 ms-2 mt-5" style="max-width: 400px; height:250px" draggable=true @dragstart="this.$parent.dragcard" :id="list.list_id+','+card.card_title" droppable="false">	
            <div class="row g-0 h-120">	
            <div class="col-md-3">	
            <img src="/static/checklist.png" class="img-fluid rounded-start" align = "left" width = "80%" style = "margin:60px 10px 3px; ">	
            </div>	
            <div class="col-md-9" style = "height:220px">	
            <div class="card-body">	
          
            <!--Card body-->	
            <h5><div class = "card-title text-wrap mt-2" :id="list.list_id+','+card.card_title" @focusout="cardChangename" contenteditable="true">{{card.card_title}}</div></h5>	
            <p class="card-content text-wrap" :id="list.list_id+','+card.card_title" @keypress="checkchar" @focusout="cardChangecontent" contenteditable="true">{{card.card_content}} </p>	
            
            <!-- Checkbox to mark the card complete-->	
            <div v-if="card.status=='Incomplete'">	
            <span :id="list.list_id+','+card.card_title">	
            <p class = "card-text mark-complete"><input type="checkbox" id="status" name="status" @click="mark_complete">&nbsp;&nbsp;Mark Complete</p></span>	
            </div>	
            <!--Link to revert to card incomplete status-->	
            <div v-else-if="card.status=='Complete'">	
            <div class = "card-text"><div class = "text-success">	
            <span v-html="tick"></span>Complete &nbsp;<span class = "card-text text-center"><a href="#"><i class="fa fa-undo" @click="mark_incomplete" :id="list.list_id+','+card.card_title"></i></a></span>
            </div>
            </div>	
            </div>	
            </div>	
            <!--Bottom half-->	
            
            </div>
                
            <!--Footer row-->	
            <div class="col align-self-start">	
            <p class="card-text text-secondary"><small class="footer-left" style = "margin:10px">Last modified: {{formatdate(card.card_due_date)}}</small></p></div>	
                
            <div class="col align-self-center">	
            <p class='card-text' :class="change_color(card.card_due_date,card.status,card.completed_date)">
            <small class="footer-right" style = "margin:10px" :id="list.list_id+','+card.card_title" @focusout="cardchangedate" contenteditable="true">Due: {{formatdate(card.card_due_date)}}</small></p></div>	
            <!--Icon to delete card-->	
            <div class="col align-self-end">	
            <p class="card-text"><a href=# v-on:click="deletecard(list.list_id,card.card_title)"><span class="delcardicon" v-html="del_icon"></span></a><small>Delete</small></p></div>	
                
            </div>	
            </div>`
        
         }

export default card

