using System;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;

namespace TextToMotionWeb.Models
{
    public class VideoTag
    {
      [Key]
      public int Id {get; set;}

      public int VideoId {get; set;}
      public Video Video {get; set;}

      public int TagId {get; set;}
      public Tag Tag {get; set;}

      public int TimeStart {get; set;}
      public int TimeEnd {get; set;}
    }
}
