import navcomp from "./navigation.js"
import maincomp from "./homepage.js"
import card from "./card.js"
import {add_list} from "./svg.js"
import {addlistlink} from "./addlist.js"
import login from "./login.js"
import register from "./register.js"
import addcard from "./add_card.js"
import summarycomp from "./summary.js"
import profile from "./profile.js"

Vue.component('navig', navcomp)
const main = Vue.component('main_data', maincomp)
Vue.component('card-template',card)
const addlistform = Vue.component('addlist',addlistlink)
const authcomp=Vue.component('auth',login)
const reg=Vue.component('register',register)
const add_card=Vue.component('add-card',addcard)
const summary=Vue.component('allsummaries',summarycomp)
const userprofile = Vue.component('userprofile',profile)

const router=new VueRouter({
    routes: [{path: '/create_list',component: addlistform, name:'createlist'},
    {path:'/',component:main,name:'home'},
    {path:'/login',component:authcomp,name:'login'},{path:'/register',component:reg,name:'register'}, 
    {path:'/addcard/:listid',component:add_card,name:"add_card", props:true},
    {path:'/summary',component:summary,name:"summary"},{path:'/profile',component:userprofile}]
})

router.beforeEach((to,from,next)=>{
    if (!localStorage.getItem('token')&&(to.path!="/login"&&to.path!="/register")){
        next({name:'login'})
    }
    else if (localStorage.getItem('token')&&(to.path=="/login"||to.path=="/register")){
        next({name:'home'})
    }
    else{
        next()
    }
})
const store=new Vuex.Store(
    {state:{
        navbar:false,
        vxcards: [],
        vxlists: [],
        username: '',
        email: '' 
    }

, mutations: {
    setlist(state,newlist){
        state.vxlists = newlist
    },
    setcard(state, newcards){
        state.vxcards = newcards
    },
}}
)


var app= new Vue({
    el: "#app",
    data: {
        lists: [],
        rows3:0,
        row_rem:0,
        cards: {},
        add_list:add_list
    },
    mounted: async function(){
        if (localStorage.getItem('token')){
            store.state.navbar=true;
        await fetch("/api/user",{'headers':{'Authentication-Token':localStorage.getItem('token')}})
        .then(res=>res.json())
        .then(data=>{store.state.username=data.Username,store.state.email=data.email})
        };  
    },
    router:router,
    store:store,
    computed: {
        navbar_comp(){return store.state.navbar},
        user(){return store.state.username}
    },
    delimiters: ['${','}$']
    })