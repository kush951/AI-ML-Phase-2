
async function match(){

    const resume = document.getElementById("resume").value.split(",");
    const jd = document.getElementById("jd").value.split(",");

    const response = await fetch("http://127.0.0.1:8000/match",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            resume_skills:resume,
            jd_skills:jd
        })
    });

    const data = await response.json();

    document.getElementById("result").innerHTML =
        "<h2>Match Score: " + data.match_score + "%</h2>";
}
