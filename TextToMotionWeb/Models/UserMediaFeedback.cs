using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using System;

namespace TextToMotionWeb.Models
{
    public class UserMediaFeedback
    {
        [Key]
        public int Id { get; set; }

        public string UserId { get; set; }
        public ApplicationUser User { get; set; }

        public int MediaId { get; set; }
        public Media Media { get; set; }

        public int Accuracy { get; set; }
        public string Message { get; set; }

        public DateTime Inserted { get; set; }
    }
}
