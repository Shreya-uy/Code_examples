let addlistlink = {
  methods:{
    addlist(event){
      event.preventDefault();
      var formdata = new FormData(event.target);
      var obj = {}

      for (const item of formdata.entries()){
          let a = item[0]
          let b= item[1]
          obj[a]=b
      }

      var postdata = JSON.stringify(obj)

      fetch("/api/lists",{headers:{'Content-Type':"application/json",'Authentication-Token':localStorage.getItem('token')},method:"POST",body:postdata})
      .then(res=>res.json())
      .then(data=>
        {if (data.message=="failure"){
          alert("List already exists. Please use a different list name")
        }
          this.$router.push("/")
        }
        )
  }
  },
template: `<form id="createlistform" @submit="addlist">
<div class = "container">
<div class="row justify-content-center">
<div class="col-md-5">
     <label for = "name" style = "margin:8px 90px"> List name </label>
     <input type = "text" class = "form-control mb-3" id = "name" name = "listname" required> 
     <label for = "desc" style = "margin:8px 120px"> Task description </label>
     <input type = "text" class = "form-control mb-3" id = "desc" name = "listdesc"  style = "height:200px">
     <input type = "submit" value = "Add" style = "margin:8px 220px">
     </div>
   </div>
 </div>
 </form>`
}
  
export {addlistlink}
