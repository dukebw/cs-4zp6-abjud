using System;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;

namespace TextToMotionWeb.Models
{
    public class Video
    {
      [Key]
      public int Id {get; set;}

      public int MediaId {get; set;}
      public Media Media {get; set;}

      public int Runtime {get; set;}
      
      public List<VideoTag> VideoTags {get; set;}
    }
}
