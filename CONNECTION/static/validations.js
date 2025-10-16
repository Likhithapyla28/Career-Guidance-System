document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("form").addEventListener("submit", function(event) {
        let isValid = true;
        
        // Validate Student Name
        let studentName = document.getElementById("student_name").value;
        if (studentName.trim() === "") {
            alert("Student Name is required.");
            isValid = false;
        }
        
        // Validate Age
        let age = document.getElementById("age").value;
        if (age < 1 || age > 100 || isNaN(age)) {
            alert("Please enter a valid age between 1 and 100.");
            isValid = false;
        }
        
        // Validate Email
        let email = document.getElementById("email").value;
        let emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailPattern.test(email)) {
            alert("Please enter a valid email address.");
            isValid = false;
        }
        
        // Validate Phone Number
        let phone = document.getElementById("phone_number").value;
        let phonePattern = /^[0-9]{10}$/;
        if (!phonePattern.test(phone)) {
            alert("Please enter a valid 10-digit phone number.");
            isValid = false;
        }
        
        // Validate SSC, Inter Score, and CGPA
        let sscScore = document.getElementById("ssc_score").value;
        let interScore = document.getElementById("inter_score").value;
        let cgpa = document.getElementById("current_cgpa").value;
        let scorePattern = /^\d+(\.\d+)?$/;
        
        if (sscScore !== "" && !scorePattern.test(sscScore)) {
            alert("Please enter a valid SSC score.");
            isValid = false;
        }
        
        if (interScore !== "" && !scorePattern.test(interScore)) {
            alert("Please enter a valid Inter score.");
            isValid = false;
        }
        
        if (cgpa !== "" && (!scorePattern.test(cgpa) || cgpa > 10)) {
            alert("Please enter a valid CGPA (up to 10). ");
            isValid = false;
        }
        
      
        
        if (!isValid) {
            event.preventDefault();
        }
    });
});
