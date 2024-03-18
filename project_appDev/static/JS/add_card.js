let addcard = {
data: 
  function() {return {current_date:new Date().toJSON().slice(0,10)}}
,
props: ['listid'],
methods: {
  addcard(event) {
    event.preventDefault()
    var formdata = new FormData(event.target)
    var obj = {}

    for (const item of formdata.entries()){
      let a = item[0]
      let b = item[1]
      obj[a]=b
    }
    var postdata = JSON.stringify(obj) 

    let lid = this.$route.params.listid
    fetch("/api/cards/"+lid,{headers:{'Content-Type':'application/json','Authentication-Token':localStorage.getItem('token')},method:"POST",body:postdata})
    .then(res=>res.json())
    .then(data=>{
      if (data.card_title==null){
        alert("Card already exists")
      }
        this.$router.push("/")
    })
  }
},  
template:` 
<form @submit="addcard">
  <div class = "container">
  <div class="row justify-content-center">
    <div class="col-md-5">
      <label for = "name" style = "margin:8px 90px"> Card Title </label>
      <textarea type = "text" class = "form-control mb-3" id = "name" name = "cardname" maxlength = "21" placeholder = "Max 21 characters" required></textarea> 
      <label for = "desc" style = "margin:8px 120px"> Content </label>
      <textarea type = "text" class = "form-control mb-3" id = "desc" name = "carddesc" style = "height:200px" maxlength="150" placeholder = "Max 150 characters"></textarea>
      <label for = "desc" style = "margin:8px 120px"> Due by </label>
      <input type = "date" class = "form-control mb-3" id = "date" name = "carddue" required :min=current_date>
      <input type = "submit" value = "Add" style = "margin:8px 220px">
      </div>
    </div>
  </div>
</form>`}

export default addcard