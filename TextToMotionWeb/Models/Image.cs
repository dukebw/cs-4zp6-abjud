using System;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;

namespace TextToMotionWeb.Models
{
  public class Image
  {
    [Key]
    public int Id {get; set;}

    public int MediaId {get; set;}
    public Media Media {get; set;}

    public string Original_image {get; set;}
    public string Processed_image { get; set; }
    public string Joint_positions {get; set;}

    public List<ImageTag> ImageTags {get; set;}
  }
}
