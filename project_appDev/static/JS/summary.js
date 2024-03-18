var summarycomp = {
    mounted:function(){this.fetchsummary()},
    methods: {
        fetchsummary: async function(){
            await fetch("/api/listsummary",{headers:{'Content-Type':'application/json','Authentication-token':localStorage.getItem('token')}})
            .then(res=>res.blob())
            .then(
                data=>{
                    let url = URL.createObjectURL(data)
                    let elem = document.getElementById("listimage")
                    elem.src=url
                }
            )
            await fetch("/api/cardsummary",{headers:{'Content-Type':'application/json','Authentication-token':localStorage.getItem('token')}})
            .then(res=>res.blob())
            .then(
                data=>{
                    let url = URL.createObjectURL(data)
                    let elem = document.getElementById("cardimage")
                    elem.src=url
                }
            )
            await fetch("/api/completionsummary",{headers:{'Content-Type':'application/json','Authentication-token':localStorage.getItem('token')}})
            .then(res=>res.blob())
            .then(
                data=>{
                    let url = URL.createObjectURL(data)
                    let elem = document.getElementById("completionimage")
                    elem.src=url
                }
            )
            await fetch("/api/lastupdatedsummary",{headers:{'Content-Type':'application/json','Authentication-token':localStorage.getItem('token')}})
            .then(res=>res.json())
            .then(
                data=>{
                    let tabledata = JSON.parse(data)
                    for (const entry in tabledata){
                    let tablebody = document.getElementById("tbody")
                    let row = tablebody.insertRow(-1)
                    let cell1 = row.insertCell(0)
                    let cell2 = row.insertCell(1)
                    cell1.innerHTML = entry
                    cell2.innerHTML = tabledata[entry]
                    }
                }
            )
        },
    },
    template: `
    <div>
    <div class="line"><div class = "fig1 element"><img id="listimage" name = "fig1" style = "margin:80px 200px 100px 200px"></div>
    <div class = "element" style="margin:80px 50px 100px 0"><p class="tableheader">Last completed card in each list</p>
    <table id="Summarytable">
    <thead>
    <td><b>List name</b></td>
    <td><b>Last card completion date</b></td>
    </thead>
    <tbody id="tbody">
    </tbody>
    </table>
    </div></div>
    <div class = "fig2"><img id="cardimage" name = "fig2" style = "margin:auto 600px"></div>
    <div class = "fig3"><img id="completionimage" name = "fig3" style = "margin:auto 600px"></div>
   
    <thead></thead>
    <br>
    <br>
    </div>
    `
}

export default summarycomp