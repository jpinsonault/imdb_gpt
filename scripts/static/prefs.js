/* static/prefs.js */
document.addEventListener("alpine:init", () => {
    Alpine.store("prefs", {
      dark: JSON.parse(localStorage.getItem("dark") || "false"),
      chartType: localStorage.getItem("chartType") || "bars",
  
      toggleDark() {
        this.dark = !this.dark;
        localStorage.setItem("dark", JSON.stringify(this.dark));
      },
      setChart(type) {
        this.chartType = type;
        localStorage.setItem("chartType", type);
      },
    });
  });
  