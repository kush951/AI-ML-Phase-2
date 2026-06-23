
async function checkMatch() {

    const studentSkills = document.getElementById("studentSkills").value.split(",");
    const requiredSkills = document.getElementById("requiredSkills").value.split(",");

    const verifiedScore = parseInt(document.getElementById("verifiedScore").value);
    const experience = parseInt(document.getElementById("experience").value);
    const minScore = parseInt(document.getElementById("minScore").value);

    const response = await fetch("http://127.0.0.1:8000/match", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            student_skills: studentSkills,
            required_skills: requiredSkills,
            verified_score: verifiedScore,
            experience: experience,
            min_score: minScore
        })
    });

    const data = await response.json();

    document.getElementById("resultCard").style.display = "block";

    document.getElementById("matchScore").innerText =
        data.match_score_percent + "%";

    document.getElementById("prediction").innerText =
        data.prediction;

    document.getElementById("lowFit").innerText =
        data.low_fit_warning;

    document.getElementById("refund").innerText =
        data.refund_triggered;

    const reasonsList = document.getElementById("reasons");
    reasonsList.innerHTML = "";

    data.reasons.forEach(reason => {
        const li = document.createElement("li");
        li.innerText = reason;
        reasonsList.appendChild(li);
    });
}
