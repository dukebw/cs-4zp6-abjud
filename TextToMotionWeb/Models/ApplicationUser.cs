using System;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;

namespace TextToMotionWeb.Models
{
    // Add profile data for application users by adding properties to the ApplicationUser class
    public class ApplicationUser : IdentityUser
    {
        public string FirstName {get; set;}
        public string LastName {get;set;}
        public string Avatar {get;set;}

        public List<Media> Media { get; set; }
        public List<UserMediaFeedback> MediaFeedback { get; set; }
    }
}
