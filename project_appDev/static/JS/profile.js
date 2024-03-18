let profile={
    data: function(){
        return{
            password:'',
            cpassword:'',
            nflag: false,
            pflag: false,
        }
    },
    computed: {
            username: function(){return this.$store.state.username},
            email: function(){return this.$store.state.email},
            passwordmatch: function(){return this.password==this.cpassword},
            passlength: function(){if (this.password.length<8){return true}}
        },
    methods: {  
        changename(){
            this.nflag = !this.nflag;
        },
        changepassword(){
            this.pflag = !this.pflag;
        },
        async submitinfo(event){
            event.preventDefault();
            let files = document.getElementById('inputfile').files
            if (files.length!=0){
                var file = files[0]
                let object = new FormData()
                object.append('file',file)
                await fetch("api/importcsv",{'headers':{'Authentication-Token':localStorage.getItem('token')},method:"POST",body:object})
                .then(res=>res.json())
                .then(data=>alert(data.message))
            }
            
            let formdata = new FormData(event.target)
            var choice = document.getElementById('selected').value
            var obj = {'Username':'','password':'','cpassword':'','email':'','report_template':choice}

            for (const item of formdata.entries()){
                obj[item[0]]=item[1]
            }
            var postdata = JSON.stringify(obj)
            if (this.pflag&&this.passlength){
                alert("Password must be of 8 characters or more")   
            }
            else{
                if (!this.passwordmatch){
                    alert("Password and confirm password must match")}
                else{
            await fetch("/api/user",{'headers':{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"POST",body:postdata})
            .then(res=>res.json())
            .then(data=>{this.$store.state.username=data.username})
            this.$router.push('/')
        }
        }
        },
        fetchformat(){
            fetch("/api/importcsv",{'headers':{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')}}) 
            .then(res=>{return res.blob()})
            .then((obj)=>{
            const url=URL.createObjectURL(obj);
            let elem = document.createElement("a");
            elem.href = url;
            elem.download = "Sample upload report.csv";
            elem.click();
    })
        }
        },
    template: `
    <form @submit="submitinfo">
    <div class="profile">
          
        <div class="profheading"> User profile of {{username}}</div>
        <div><i class="bi bi-person-circle" style="font-size:8rem"></i></div>
        <br>
        <label class="proftext">Username : {{username}}</label>&nbsp;&nbsp;
        <button type="button" class="btn btn-outline-secondary" @click="changename">Change Username</button>
        <div>
        <div v-if="nflag==true" class="proftext">Enter new username: 
        <input type = "text" class = "profileusername" name="Username" required/></div>
        </div>
        
        <div class="proftext">User email: {{email}} 
        </div>
          
        <div><button type="button" class="btn btn-outline-secondary" @click="changepassword">Change Password</button></div>

        <label v-if="pflag==true" class="proftext">New Password</label> 
        <div><input v-if="pflag==true" v-model="password" type="password" class="profilepassword" name="password" required/></div>
        <label v-if="pflag==true" class="proftext">Confirm New Password</label> 
        <div><input v-if="pflag==true" v-model="cpassword" type="text" class="profilepassword" name="cpassword" required/></div>
        <div v-if="passwordmatch==false" class="text-danger">Passwords do not match</div>
        <div v-if="pflag&&passlength==true" class="text-danger">Password must be of 8 characters or more</div>
        
        <div class="proftext"><span>Email report template:</span><div class="dropdown email_block">
        
        <select class="form-select selectreport border border-secondary" id="selected" aria-label="Default select example">
        <option value="HTML">HTML</option>
        <option value="PDF">PDF</option>
        </select>
        </div></div>

        <div class="proftext"><span>Upload dashboard tasks from csv file: </span><div class="uploadfile"><input type="file" class="form-control" id="inputfile"></div></div>
        <div><button type="button" class="btn btn-outline-primary" @click="fetchformat">Sample csv report format for upload</button></div>

        <button class="btn btn-secondary btn-lg active" role="button" aria-pressed="true">Save</button>
    </div>
    </form>
   `
}

export default profile