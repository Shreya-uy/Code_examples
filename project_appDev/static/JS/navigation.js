let navcomp = {
    computed: {
      navbar_value(){return this.$store.state.navbar}
    },
    methods: {
      logout(){
      localStorage.removeItem('token');
      this.$store.state.navbar=false
      this.$router.push("/login")
    },
    exportcsv(){
     fetch("/api/exportcsv",{headers:{'Content-Type':'application/json','Authentication-token':localStorage.getItem('token')}})
     .then(res=>{return res.blob()})
      .then((obj)=>{
      const url=URL.createObjectURL(obj);
      let elem = document.createElement("a");
      elem.href = url;
      elem.download = "BoardIt dashboard report.csv";
      elem.click();
    });   
    }
  },
    template: `<nav v-if="navbar_value==true" class="navbar navbar-expand-lg mt-5 bg-secondary">
    <div class="container-fluid">
      <a class="navbar-brand text-warning">BoardIt</a>
      
      <ul class="navbar-nav me-auto mb-2">
      <li class="nav-item">
      <router-link to="/profile" class="nav-link active text-white" aria-current="page">Profile</router-link></li>
      <li class="nav-item">
      <router-link to="/" class="nav-link active text-white" aria-current="page">Home</router-link></li>
      <li class = "nav-item">
      <router-link to="/summary" class="nav-link active text-white" aria-current="page">Summary</router-link></li>
      </ul>
        
      <ul class="navbar-nav me-2 mb-2">
      <li class = "nav-item"></li>
      <button class="btn btn-light nav-link active" aria-current="page" @click="$router.push('/profile')">Import report</button></li>
      </ul>
      <ul class="navbar-nav me-2 mb-2">
      <li class = "nav-item"></li>
      <button class="btn btn-light nav-link active" aria-current="page" @click="exportcsv">Export report</button></li>
      </ul>
      <ul class="navbar-nav me-2 mb-2">
      <li class = "nav-item"></li>
      <a href="#" class="nav-link active text-white" aria-current="page" @click="logout">Logout</a></li>
      </ul>     
    
      </div>
    </nav>`
}

export default navcomp
