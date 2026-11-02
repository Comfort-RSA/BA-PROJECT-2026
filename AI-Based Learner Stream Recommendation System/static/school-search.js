

document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("school-search-input");
  const resultsList = document.getElementById("school-results");
  const hiddenField = document.getElementById("school-name-field");
  const selectedCard = document.getElementById("selected-school-card");
  const submitBtn = document.getElementById("school-submit-btn");

  if (!input) return; // not on this page

  let schools = [];
  let filtered = [];

  fetch("/api/schools")
    .then((res) => res.json())
    .then((data) => {
      schools = data;
    })
    .catch(() => {
      resultsList.innerHTML = "<li>Could not load the school list. Please refresh.</li>";
    });

  function renderResults(list) {
    resultsList.innerHTML = "";
    if (list.length === 0) {
      resultsList.classList.remove("open");
      return;
    }
    list.slice(0, 20).forEach((school) => {
      const li = document.createElement("li");
      li.innerHTML =
        '<div class="school-name">' + school.name + "</div>" +
        '<div class="school-meta">' + school.municipality + ", " + school.province + "</div>";
      li.addEventListener("click", () => selectSchool(school));
      resultsList.appendChild(li);
    });
    resultsList.classList.add("open");
  }

  function selectSchool(school) {
    hiddenField.value = school.name;
    input.value = school.name;
    resultsList.classList.remove("open");

    document.getElementById("selected-school-name").textContent = school.name;
    document.getElementById("selected-school-location").textContent =
      school.municipality + ", " + school.province;

    const subjectsWrap = document.getElementById("selected-school-subjects");
    subjectsWrap.innerHTML = "";
    school.subjects.forEach((subj) => {
      const span = document.createElement("span");
      span.className = "subject-tag";
      span.textContent = subj;
      subjectsWrap.appendChild(span);
    });

    const streamsWrap = document.getElementById("selected-school-streams");
    streamsWrap.innerHTML = "";
    school.streams.forEach((stream) => {
      const span = document.createElement("span");
      span.className = "stream-tag";
      span.textContent = stream;
      streamsWrap.appendChild(span);
    });

    selectedCard.classList.add("visible");
    submitBtn.disabled = false;
  }

  input.addEventListener("input", function () {
    const q = input.value.trim().toLowerCase();
    hiddenField.value = ""; // typing invalidates a previous selection
    submitBtn.disabled = true;
    selectedCard.classList.remove("visible");

    if (q.length === 0) {
      resultsList.classList.remove("open");
      return;
    }
    filtered = schools.filter(
      (s) =>
        s.name.toLowerCase().includes(q) ||
        s.municipality.toLowerCase().includes(q) ||
        s.province.toLowerCase().includes(q)
    );
    renderResults(filtered);
  });

  input.addEventListener("focus", function () {
    if (input.value.trim().length > 0 && filtered.length > 0) {
      resultsList.classList.add("open");
    }
  });

  document.addEventListener("click", function (e) {
    if (!e.target.closest(".school-search-wrap")) {
      resultsList.classList.remove("open");
    }
  });
});
