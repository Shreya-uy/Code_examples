import {remove, edit, add_card,add_list,export_list} from "./svg.js"
import card from "./card.js"

let maincomp = {
    data: function(){
        return {
        remove: remove,
        edit: edit,
        add_card: add_card,
        tempcards:[],
        draggeddata:"",
        add_list:add_list,
        export_list: export_list
    }},
    mounted: async function() {
        await this.get_lists();
        await this.get_cards();
        
        this.$store.commit('setcard',this.cards);},

    computed: {
        list_count: function() {
            let ll=this.$store.state.vxlists.length
            return ll
        },
        lists : function(){
            return this.$store.state.vxlists
        },
        cards: function() {
            return this.$store.state.vxcards
        }
    },
       methods:{
        get_lists: async function(){
            await fetch("/api/lists",{'headers':{'Authentication-Token':localStorage.getItem('token')}})
            .then(response=>response.json())
            .then(data => {this.$store.commit('setlist',data); this.$store.state.vxlists.forEach(item=>this.$set(this.tempcards,item.list_id,[]))})
            .catch(err=>console.log(err));
            },  
            get_cards: async function(){
                await fetch("/api/cards",{'headers':{'Authentication-Token':localStorage.getItem('token')}})
                .then(response=> response.json())
                .then(data=>{data.forEach(card=>{this.tempcards[card.list_id].push(card)}),this.$store.commit('setcard',this.tempcards)})
                .catch(err=>console.log(err)
                )},
        listChangename (event) {    
            let obj = {}
            obj['listname']=event.target.textContent
            obj['listdesc']=this.lists[event.target.id].description
            let lid = this.lists[event.target.id].list_id
            let edit_obj = JSON.stringify(obj)
            fetch("/api/lists/"+lid,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"PUT",body:edit_obj})
            .then(resp=>resp.json())
            .then(data=> {if (data.message=="failure"){event.target.innerText=data.listname; alert("List already exists, use a different name")} 
            else {this.lists[event.target.id].list_name=data.listname}})
        },
        listChangedesc (event) {
            let obj = {}
            obj['listname']=this.lists[event.target.id].list_name
            let lid = this.lists[event.target.id].list_id
            obj['listdesc']=event.target.textContent
            let edit_obj = JSON.stringify(obj)
            fetch("/api/lists/"+lid,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"PUT",body:edit_obj})
            .then(resp=>resp.json())
            .then(data=> {if (data.message=="failure"){event.target.innerText=data.description} 
            else {this.lists[event.target.id].description=data.description}})
        },
        
        deletelist(index){
            let lid=this.lists[index].list_id
            fetch("/api/lists/"+lid,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"DELETE"})
            this.lists.splice(index,1);
        },
          dragcard(event){
            this.draggeddata = event.target;
          },
        
          stopdrag(event){
            event.preventDefault();
          },
          dropcard(event){
            event.preventDefault();
            var elem = event.target
            
            if (!(elem.classList.contains('draggableclass'))){
            while (!(elem.classList.contains('list-content'))) {
            elem = elem.parentElement
        }}   
            else{
                for (let child of elem.children){
                    elem = child}
            }        
            
            var new_lid = elem.id
            var dragged_lid = this.draggeddata.id.split(',')[0]
            var dragged_ctitle =  this.draggeddata.id.split(',')[1]

            
            fetch("/api/cards/"+dragged_lid+"/"+dragged_ctitle+"/"+new_lid,{headers:{"Content-Type":"application/json",'Authentication-Token':localStorage.getItem('token')},method:"PATCH"})
            .then(res=>res.json())
            .then((data)=>
                {if (data.message=="failure"){
                alert("Cannot move as card with title already exists in target list")}
                else{
                    var card = this.cards[dragged_lid].filter((card)=>card.card_title==dragged_ctitle)[0]
                    if (dragged_lid != new_lid){
                    this.cards[new_lid].push(card)
                    let newcardslist = this.cards[dragged_lid].filter((card)=>card.card_title!=dragged_ctitle)
                    this.$set(this.cards,dragged_lid,newcardslist)
                    }  
                }
            })
          },
          exportcards(event){
            let formdata = new FormData()
            let listid = event.target.parentElement.parentElement.id
            formdata.append('listid',listid)
            fetch("/api/exportcsv",{headers:{'Authentication-Token':localStorage.getItem('token')},method:"POST",body:formdata})
            .then(res=>res.blob())
            .then(
                data=>{
                    let url = URL.createObjectURL(data)
                    let elem = document.createElement('a')
                    elem.href = url
                    elem.download = "BoardIt list report _"+listid+".csv" 
                    elem.click()
                }
            )

          }
    },
template: `
    <div>
        <div class="addlisticon"><router-link to="/create_list"><span v-html="add_list"></span></router-link>&nbsp;Add a list</div>
        <div class="row m-3 justify-content-center">
        <div v-for="index in list_count" v-bind:key="index" class="col-3 collist border border-4 bg-light m-4 draggableclass" @dragover="stopdrag" @drop="dropcard">
        
        <div class="list-content" :id="lists[index-1].list_id"><b>
        
        <!-- List name and content-->
        <div class = "list-name" :id="index-1" @focusout="listChangename" contenteditable="true">{{lists[index-1].list_name}}</div></b>
        <div class = "desc sub-col p-4 text-wrap" :id="index-1" @focusout="listChangedesc" contenteditable="true">{{lists[index-1].description}}</div>
        
        <div id = "modifyicon" class="text-center">
        <a id = "modifylist" data-bs-toggle="modal" :data-bs-target="'#popup_'+lists[index-1].list_id" style="color:maroon">
        <span v-html="remove"></span>
        Delete list
        </a>&nbsp;
        
        <router-link :to="'/addcard/'+lists[index-1].list_id">
        <span v-html="add_card"></span>
        </router-link> 
        <span id = "modifylist">Add card</span>

        <a href=# @click="exportcards" :id="lists[index-1].list_id"><span class="exportlisticon" v-html="export_list"></span></a>
        Export as csv
        
        <!--Modal for deleting list-->
        
        <div class="modal fade" v-bind:id="'popup_'+lists[index-1].list_id" tabindex="-1" aria-labelledby="deleteModal" aria-hidden="true">
        <div class="modal-dialog">
        <div class="modal-content">
        <div class="modal-header">
        <h5 class="modal-title" id="deleteModal">Alert</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body text-wrap text-center">
        Deleting list {{lists[index-1].list_name}} will delete all the associated cards. Are you sure you want to delete?
        </div>
        <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-primary" data-bs-dismiss="modal" v-on:click="deletelist(index-1)">Delete</button>
        </div>
        </div>
        </div>
        </div>
        
        </div>
        <!--Nested loop for cards--> 

        <card-template :list="lists[index-1]" v-for="card in cards[lists[index-1].list_id]" :key="lists[index-1].list_id+card.card_title" :card="card">
        </card-template>

        </div>

        

        </div>
        <span id="endofpage"></span>   
        </div>
    </div>    `
        ,
}

export default maincomp

