
function predictMatch() {

  const python = parseInt(document.getElementById("python").value);
  const js = parseInt(document.getElementById("javascript").value);
  const sql = parseInt(document.getElementById("sql").value);
  const aws = parseInt(document.getElementById("aws").value);
  const gpa = parseFloat(document.getElementById("gpa").value);

  let score = (python + js + sql + aws) / 4;

  if (gpa > 8) {
    score += 5;
  }

  if (score > 100) {
    score = 100;
  }

  const isMatch = score >= 75;

  document.getElementById("resultBox").style.display = "block";

  const badge = document.getElementById("matchBadge");

  if (isMatch) {
    badge.innerHTML = "MATCH";
    badge.style.backgroundColor = "#28a745";
  } else {
    badge.innerHTML = "NO MATCH";
    badge.style.backgroundColor = "#dc3545";
  }

  document.getElementById("confidenceBar").style.width = score + "%";
  document.getElementById("confidenceBar").style.backgroundColor = "#667eea";

  document.getElementById("confidenceText").innerHTML =
    score.toFixed(1) + "%";

  document.getElementById("explanation").innerHTML =
    "The candidate matches based on verified skills, GPA, and overall technical profile.";

  const recommendations = document.getElementById("recommendations");

  recommendations.innerHTML = `
    <li>Improve AWS skills</li>
    <li>Complete more projects</li>
    <li>Maintain strong coding profile</li>
  `;
}

