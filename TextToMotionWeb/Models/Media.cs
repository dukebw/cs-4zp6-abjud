using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using System;

namespace TextToMotionWeb.Models
{
    public class Media
    {
      [Key]
      public int Id {get; set;}
      //important FK
      public string UserId {get; set;}
      public ApplicationUser User {get; set;}

      public string Name {get; set;}
      public string Caption {get; set;}
      public string Category {get; set;}
      public string Type {get; set;}
      public string Group {get; set;}

      public DateTime Inserted {get; set;}
      public DateTime Updated {get; set;}

      //relations
      public List<UserMediaFeedback> UserFeedback {get; set;}

      public Image Image {get; set;}
      public Video Video {get; set;}

    }

}
