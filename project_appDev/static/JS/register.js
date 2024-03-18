let register = {
    data: function(){
        return {
            password: "",
            confirmpassword: "",
        }
    },
    methods: {
        async register(event){
            event.preventDefault();

            const formdata = new FormData(event.target);
            var postdata = new URLSearchParams();

            for (var item of formdata.entries()) {
               postdata.append(item[0],item[1])
            }

            await fetch("/register",{headers:{"Content-Type":"application/x-www-form-urlencoded","Accept":"application/json"},method:"POST",body: postdata})
            .then(res=>res.json())
            .then(
                data=>{
               if (data.meta.code != 200){
               if ("email" in  data.response.field_errors)
               {alert(data.response.field_errors.email)}
               if ("password" in data.response.field_errors)
               {alert(data.response.field_errors.password)}
               }
               else{
                alert("Registration successful");
                this.$router.push("/login");
               }
            })
            .catch((err)=>console.log(err));
            
        }},
        computed: {
            passwordmatch: function() {
                if (this.password==this.confirmpassword){
                    return false;
                }
                return true
            }
        },
    template: ` <div>
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
                <form @submit="register">
                        <br><br>
                    <div class = "form-heading">Register</div><br>
                    <div class = "form_element">
                    <div class="my-3 px-3"></div>
                        <label for="username">Username</label> <input class="form-control" id="username" name="Username" value=""> 
							</div>
                        <div class = "form_element">
                        <div class="my-3 px-3"></div>
                        <label for="email">Email Address</label> <input class="form-control" id="exampleInputEmail1" name="email" required type="email" value="">
                        </div>
                        <div class = "form_element">
                        <div class="my-3 px-3"></div>
                        <label for="password">Password</label> <input class="form-control" id="password" name="password" required type="password" value="" v-model="password"> 
                        </div>
                        <div class = "form_element">
                        <div class="my-3 px-3"></div>
                        <label for="password">Confirm Password</label> <input class="form-control" id="confirmpassword" name="password_confirm" required type="text" value="" v-model="confirmpassword"> 
                        <small style="color:red"><div v-if="passwordmatch"> Passwords do not match </div> </small>
                        </div>

                        <div class = "form_element">
                        <div class="my-3 px-3"></div></div>
                        <div class = "form_element">
                        <div class="my-3 px-3"></div></div>
                        <div class = "form_element"><input type="submit" id = "submit" value = "Register"></div><br>
                        <div class = "form_element"><div class = "form_element">Go back to <span><router-link to="/login">login</router-link></span></div></div>
                        </form>

                        </div>
                    
                </div>
            </div>
        </div>`
}

export default register
