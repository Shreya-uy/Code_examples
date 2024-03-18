let login={
    mounted: function(){document.title = "BoardIt"},
    methods: {
        authenticate(event){
            event.preventDefault();
            const formdata = new FormData(event.target);
            var form_obj = {}
            var obj = {}
            for (var item of formdata.entries()) {
               let a = item[0]
               let b = item[1]
               obj[a]=b
            }
           
            form_obj = JSON.stringify(obj)

            fetch("/login?include_auth_token",{headers:{"Content-Type":"application/json"},method:"POST",body: form_obj})
            .then(res=>{return res.json()})
            .then(data=>{
                console.log(data)
                if (data.meta.code!=200){
                if (('password' in data.response.field_errors)){
                    alert("Please enter valid password")
                }
                else if (('email' in data.response.field_errors)){
                    alert("User not registered")
                }
            }
                else{
                localStorage.setItem('token',data.response.user.authentication_token),this.$router.push("/");
                this.$store.state.navbar=true;
                fetch("/api/user",{'headers':{'Authentication-Token':localStorage.getItem('token')}})
                .then(res=>res.json())
                .then(data=>{this.$store.state.username=data.Username,this.$store.state.email=data.email})
            }
            })
            .catch((err)=>console.log(err))
            
        },
    },
    template: `
    <div>
    <h1>BoardIt</h1>
    <div class = "page_title">Welcome to BoardIt</div>
	<br>
	<br>
	<div class = "first_line"> BoardIt is a Platform that allows you to </div>
	<div class = "list">
		<ul>
			<li>organise your tasks in your own personal dashboard</li>
			<li>customise and add cards for each task</li>
			<li>keep track of completed and pending tasks</li>
		</ul>
	</div>
	<div class = "login-box">
		<div class="container">
			<div class="row justify-content-center">
				<div class="col-md-8">
					<form @submit="authenticate">
							<br><br>
						<div class = "form-heading">Login</div><br>
						<div class = "form_element">
                        <div class="my-3 px-3"></div>
                            <label for="email">Email Address</label> <input class="form-control" id="email" name="email" required type="email" value=""> 
                            </div>
						<div class = "form_element">
						<div class="my-3 px-3"></div>
                        <label for="password">Password</label> <input class="form-control" id="password" name="password" required type="password" value=""> 
                        </div>
						<div class = "form_element"><div class="my-3 px-3">
                        <p><input id="submit" name="submit" type="submit" value="Submit"></p>
                        </div></div>
						<div class = "form_element"><div class = "form_element">New User? Register <span><router-link to="/register">here</router-link></span></div></div>
					</form>
				</div>
			</div>
		</div>
    </div>
</div>`
}

export default login