using System;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;

namespace TextToMotionWeb.Models
{
    public class ImageTag
    {
      [Key]
      public int Id {get; set;}

      public int ImageId {get; set;}
      public Image Image {get; set;}

      public int TagId {get; set;}
      public Tag Tag {get; set;}
    }
}
