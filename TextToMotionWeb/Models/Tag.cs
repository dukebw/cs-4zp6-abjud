using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using System;

namespace TextToMotionWeb.Models
{
    public class Tag
    {
      [Key]
      public int Id {get; set;}

      public string Name {get; set;}
      public string Classification {get; set;}

      public DateTime Inserted {get; set;}

      //relations 
      public List<ImageTag> ImageTags {get; set;}
      public List<VideoTag> VideoTags {get; set;}
    }
  }
